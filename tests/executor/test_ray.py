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

    def test_dependencies(self):
        """Test that multiple executors can depend on the same parent executor and the parent only runs once."""

        parent_log_dir = get_datafolder(f"{self.tmp_dir}/parent")

        class ParentSimpleStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                with open(parent_log_dir.resolve_paths("parent.txt"), "a") as f:
                    f.write(f"called {rank}\n")

        class ChildSimpleStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                pass

        # Create parent executor
        parent_executor = RayPipelineExecutor(
            pipeline=[ParentSimpleStep()],
            tasks=2,
            workers=2,
            logging_dir=parent_log_dir,
        )

        # Create two child executors that depend on the same parent
        child1_log_dir = get_datafolder(f"{self.tmp_dir}/child1")
        child1_executor = RayPipelineExecutor(
            pipeline=[ChildSimpleStep()],
            tasks=2,
            workers=2,
            logging_dir=child1_log_dir,
            depends=parent_executor,
        )

        child2_log_dir = get_datafolder(f"{self.tmp_dir}/child2")
        child2_executor = RayPipelineExecutor(
            pipeline=[ChildSimpleStep()],
            tasks=2,
            workers=2,
            logging_dir=child2_log_dir,
            depends=parent_executor,
        )

        # Run child1 - this should launch the parent first
        child1_executor.run()
        child2_executor.run()
        with open(parent_log_dir.resolve_paths("parent.txt"), "r") as f:
            # Two calls because of two tasks
            self.assertEqual(sorted(f.read().strip().splitlines()), ["called 0", "called 1"])
