#!/usr/bin/env python3
import os
import shutil
import tempfile
import unittest

import ray

from datatrove.executor.ray import RayPipelineExecutor
from datatrove.io import get_datafolder
from datatrove.utils._import_utils import is_boto3_available, is_moto_available, is_s3fs_available

# Decorators to skip tests if boto3/moto/s3fs unavailable
from ..utils import require_boto3, require_moto, require_s3fs


if is_boto3_available():
    import boto3

if is_moto_available():
    from moto.moto_server.threaded_moto_server import ThreadedMotoServer

if is_s3fs_available():
    from s3fs import S3FileSystem

port = 5555
endpoint_uri = f"http://127.0.0.1:{port}/"


@require_moto
class TestRayExecutor(unittest.TestCase):
    def setUp(self):
        # Start Moto’s local S3 mock server
        self.server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
        self.server.start()
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_ACCESS_KEY_ID"] = "foo"

        ray.init()

        # Local temp folder
        self.tmp_dir = tempfile.mkdtemp()

        # Clean up after test
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        self.addCleanup(self.server.stop)
        self.addCleanup(ray.shutdown)

    @require_boto3
    @require_s3fs
    def test_executor(self):
        """
        Test the RayPipelineExecutor with different (tasks, workers)-like configs,
        verifying that it writes the same completion/log/stat files as the Local/Slurm executors.
        """
        s3fs = S3FileSystem(client_kwargs={"endpoint_url": endpoint_uri})
        s3 = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_uri)
        s3.create_bucket(Bucket="test-bucket")

        configurations = ((3, 1), (3, 3), (3, -1))

        file_list = [
            "executor.json",
            "stats.json",
        ] + [
            x
            for rank in range(3)
            for x in (f"completions/{rank:05d}", f"logs/task_{rank:05d}.log", f"stats/{rank:05d}.json")
        ]

        for tasks, workers in configurations:
            for log_dir in (
                f"{self.tmp_dir}/logs_{tasks}_{workers}",
                (f"s3://test-bucket/logs_{tasks}_{workers}", s3fs),
            ):
                log_dir = get_datafolder(log_dir)

                executor = RayPipelineExecutor(
                    pipeline=[],
                    tasks=tasks,
                    workers=workers,
                    logging_dir=log_dir,
                )
                executor.run()

                # Verify the expected files exist
                for file in file_list:
                    self.assertTrue(
                        log_dir.isfile(file),
                        f"Expected file {file} was not found in {log_dir}",
                    )
