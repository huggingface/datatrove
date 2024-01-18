import shutil
import tempfile
import unittest

import boto3
from moto.moto_server.threaded_moto_server import ThreadedMotoServer
from s3fs import S3FileSystem

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import get_datafolder


EXAMPLE_DIRS = ("/home/testuser/somedir", "file:///home/testuser2/somedir", "s3://test-bucket/somedir")
FULL_PATHS = (
    "/home/testuser/somedir/file.txt",
    "/home/testuser2/somedir/file.txt",
    "s3://test-bucket/somedir/file.txt",
)


port = 5555
endpoint_uri = "http://127.0.0.1:%s/" % port


class TestLocalExecutor(unittest.TestCase):
    def setUp(self):
        self.server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
        self.server.start()

        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        self.addCleanup(self.server.stop)

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
