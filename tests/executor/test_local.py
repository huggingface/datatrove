import json
import os
import shutil
import tempfile
import unittest

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils._import_utils import is_boto3_available, is_moto_available, is_s3fs_available

from ..utils import require_boto3, require_moto, require_s3fs


EXAMPLE_DIRS = ("/home/testuser/somedir", "file:///home/testuser2/somedir", "s3://test-bucket/somedir")
FULL_PATHS = (
    "/home/testuser/somedir/file.txt",
    "/home/testuser2/somedir/file.txt",
    "s3://test-bucket/somedir/file.txt",
)


port = 5555
endpoint_uri = "http://127.0.0.1:%s/" % port


if is_boto3_available():
    import boto3  # noqa: F811

if is_moto_available():
    from moto.moto_server.threaded_moto_server import ThreadedMotoServer  # noqa: F811

if is_s3fs_available():
    from s3fs import S3FileSystem  # noqa: F811


class DocGenerator(PipelineStep):
    """Yields a few documents, tracking a "docs" stat for each one."""

    def __init__(self, docs_per_rank: int):
        super().__init__()
        self.docs_per_rank = docs_per_rank

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc_i in range(self.docs_per_rank):
            self.stat_update("docs")
            yield Document(text=f"document {doc_i}", id=f"{rank}_{doc_i}")


class FailFirstTime(PipelineStep):
    """Raises the first time it is called for fail_rank, succeeds when relaunched."""

    def __init__(self, marker_dir: str, fail_rank: int):
        super().__init__()
        self.marker_dir = marker_dir
        self.fail_rank = fail_rank

    def run(self, data, rank: int = 0, world_size: int = 1):
        marker_file = os.path.join(self.marker_dir, str(rank))
        if rank == self.fail_rank and not os.path.isfile(marker_file):
            open(marker_file, "w").close()
            raise RuntimeError(f"Simulated failure for {rank=}")
        yield from data


class TestLocalExecutorStats(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_relaunched_job_stats_include_previously_completed_tasks(self):
        tasks, docs_per_rank, fail_rank = 4, 5, 2
        logging_dir = os.path.join(self.tmp_dir, "logs")
        marker_dir = os.path.join(self.tmp_dir, "markers")
        os.makedirs(marker_dir)

        def get_executor():
            return LocalPipelineExecutor(
                pipeline=[DocGenerator(docs_per_rank), FailFirstTime(marker_dir, fail_rank)],
                tasks=tasks,
                workers=1,
                logging_dir=logging_dir,
            )

        # first run: fail_rank fails, ranks after it never run
        with self.assertRaises(RuntimeError):
            get_executor().run()
        # relaunch: only the incomplete ranks are rerun, but the merged stats should still cover all tasks
        stats = get_executor().run()

        with get_datafolder(logging_dir).open("stats.json", "rt") as f:
            saved_stats = json.load(f)
        for stats_dict in (json.loads(stats.to_json()), saved_stats):
            docs_stat = stats_dict[0]["stats"]["docs"]
            # metric stats with a single relevant field are serialized as just their total
            docs_total = docs_stat["total"] if isinstance(docs_stat, dict) else docs_stat
            self.assertEqual(docs_total, tasks * docs_per_rank)


@require_moto
class TestLocalExecutor(unittest.TestCase):
    def setUp(self):
        self.server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
        self.server.start()
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_ACCESS_KEY_ID"] = "foo"

        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        self.addCleanup(self.server.stop)

    @require_boto3
    @require_s3fs
    def test_executor(self):
        s3fs = S3FileSystem(client_kwargs={"endpoint_url": endpoint_uri})
        s3 = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_uri)
        s3.create_bucket(Bucket="test-bucket")
        configurations = (3, 1), (3, 3), (3, -1)
        file_list = [
            "executor.json",
            "stats.json",
        ] + [
            x
            for rank in range(3)
            for x in (f"completions/{rank:05d}", f"logs/task_{rank:05d}.log", f"stats/{rank:05d}.json")
        ]
        for tasks, workers in configurations:
            for log_dir in (f"{self.tmp_dir}/{tasks}_{workers}", (f"s3://test-bucket/logs/{tasks}_{workers}", s3fs)):
                log_dir = get_datafolder(log_dir)
                executor = LocalPipelineExecutor(pipeline=[], tasks=tasks, workers=workers, logging_dir=log_dir)
                executor.run()

                for file in file_list:
                    assert log_dir.isfile(file)
