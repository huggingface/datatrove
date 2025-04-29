#!/usr/bin/env python3
import shutil
import tempfile
import unittest

import ray

from datatrove.executor.ray import RayPipelineExecutor
from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep


port = 5555
endpoint_uri = f"http://127.0.0.1:{port}/"


class TestRayExecutor(unittest.TestCase):
    def setUp(self):
        ray.init()

        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        self.addCleanup(ray.shutdown)

    def test_executor(self):
        configurations = ((3, 1), (3, 3), (3, -1))

        file_list = [
            "executor.json",
            "stats.json",
        ] + [
            x
            for rank in range(3)
            for x in (f"completions/{rank:05d}", f"logs/task_{rank:05d}.log", f"stats/{rank:05d}.json")
        ]

        class SleepBlock(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                for i in range(10):
                    yield i

        for tasks, workers in configurations:
            for log_dir in (f"{self.tmp_dir}/logs_{tasks}_{workers}",):
                log_dir = get_datafolder(log_dir)

                executor = RayPipelineExecutor(
                    pipeline=[SleepBlock()],
                    tasks=tasks,
                    workers=workers,
                    logging_dir=log_dir,
                )
                executor.run()

                for file in file_list:
                    self.assertTrue(
                        log_dir.isfile(file),
                        f"Expected file {file} was not found in {log_dir}",
                    )
