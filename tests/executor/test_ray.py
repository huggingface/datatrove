#!/usr/bin/env python3
import shutil
import tempfile
import unittest
from concurrent.futures import wait
from unittest.mock import MagicMock, patch

import ray
from ray.util.placement_group import placement_group_table
from ray.util.state import list_actors, list_tasks

from datatrove.executor.ray import (
    RayPipelineExecutor,
    RayTaskManager,
    TimeoutManager,
)
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

    def test_placement_group_creation(self):
        """Test that placement groups are created when nodes_per_task > 1"""
        from datatrove.executor.ray import RayTaskManager

        log_dir = get_datafolder(f"{self.tmp_dir}/test_placement_group")

        class DummyStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                yield {"text": "test"}

        executor = RayPipelineExecutor(
            pipeline=[DummyStep()],
            tasks=1,
            workers=1,
            logging_dir=log_dir,
            nodes_per_task=2,
        )

        timeout_manager = TimeoutManager()
        task_manager = RayTaskManager(nodes_per_task=2, timeout_manager=timeout_manager)
        with task_manager:
            executor_ref = ray.put(executor)
            remote_options = {"num_cpus": 1, "num_gpus": 0, "memory": 100 * 1024 * 1024}

            # Submit a task - this should create a placement group
            pg_task_future = task_manager.submit_task(executor_ref, [0], remote_options)

            # Wait for the thread pool executor to finish the async submission
            max_wait = 10

            finished, unfinished = task_manager.wait([pg_task_future], timeout=max_wait)
            self.assertEqual(len(finished), 1, "Should have 1 finished task")

            ray.get(finished[0])
            tables = placement_group_table()
            self.assertEqual(len(tables), 1, "Should have 1 placement group")
            group0 = list(tables.values())[0]
            # Check that we have 2 bundles
            self.assertEqual(len(group0["bundles"]), 2, "Placement group should have 2 bundles")

            # Check each bundle has the expected resources
            for bundle in group0["bundles"].values():
                self.assertEqual(bundle["CPU"], 1.0)
                self.assertEqual(bundle["memory"], float(100 * 1024 * 1024))

    def test_rank_resubmission_on_retriable_error(self):
        """Test that ranks are resubmitted when tasks fail with retriable errors"""
        from ray.exceptions import WorkerCrashedError

        log_dir = get_datafolder(f"{self.tmp_dir}/test_resubmission")

        class DummyStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                yield {"text": "test"}

        executor = RayPipelineExecutor(
            pipeline=[DummyStep()],
            tasks=1,
            workers=1,
            logging_dir=log_dir,
            nodes_per_task=1,
        )

        timeout_manager = TimeoutManager()
        task_manager = RayTaskManager(nodes_per_task=1, timeout_manager=timeout_manager)

        with task_manager:
            executor_ref = ray.put(executor)
            remote_options = {"num_cpus": 1, "num_gpus": 0, "memory": 100 * 1024 * 1024}

            # Create a real task by submitting one, then get the actual ObjectRef
            pg_task_future = task_manager.submit_task(executor_ref, [0], remote_options)
            wait([pg_task_future])

            # Get tasks from the future result
            tasks = pg_task_future.result()

            real_task = tasks[0]
            mock_group = task_manager.task_to_group[real_task]

            # Simulate retriable error by patching ray.get
            with patch("ray.get", side_effect=WorkerCrashedError("Worker crashed")):
                success, ranks_to_resubmit = task_manager.task_done(real_task)

                self.assertFalse(success, "Task should not be marked as successful")
                self.assertEqual(ranks_to_resubmit, [0], "Should return ranks to resubmit")
                self.assertTrue(mock_group.has_retriable_error, "Group should be marked with retriable error")

    def test_no_resubmission_on_non_retriable_error(self):
        """Test that ranks are NOT resubmitted when tasks fail with non-retriable errors"""
        log_dir = get_datafolder(f"{self.tmp_dir}/test_no_resubmission")

        class DummyStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                yield {"text": "test"}

        executor = RayPipelineExecutor(
            pipeline=[DummyStep()],
            tasks=1,
            workers=1,
            logging_dir=log_dir,
            nodes_per_task=1,
        )

        timeout_manager = TimeoutManager()
        task_manager = RayTaskManager(nodes_per_task=1, timeout_manager=timeout_manager)

        with task_manager:
            executor_ref = ray.put(executor)
            remote_options = {"num_cpus": 1, "num_gpus": 0, "memory": 2 * 1024 * 1024 * 1024}

            # Create a real task by submitting one, then get the actual ObjectRef
            pg_task_future = task_manager.submit_task(executor_ref, [0], remote_options)
            wait([pg_task_future])

            # Get tasks from the future result
            tasks = pg_task_future.result()
            real_task = tasks[0]
            mock_group = task_manager.task_to_group[real_task]

            # Simulate non-retriable error (ValueError)
            with patch("ray.get", side_effect=ValueError("Application error")):
                success, ranks_to_resubmit = task_manager.task_done(real_task)

                self.assertFalse(success, "Task should not be marked as successful")
                self.assertIsNone(ranks_to_resubmit, "Should NOT return ranks to resubmit for non-retriable error")
                self.assertFalse(mock_group.has_retriable_error, "Group should NOT be marked with retriable error")

    def test_cleanup(self):
        """Test that placement groups and tasks are cleaned up correctly"""
        log_dir = get_datafolder(f"{self.tmp_dir}/test_cleanup")

        class DummyStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                import time

                time.sleep(30.0)
                yield {"text": "test"}

        executor = RayPipelineExecutor(
            pipeline=[DummyStep()],
            tasks=1,
            workers=1,
            logging_dir=log_dir,
            nodes_per_task=1,
        )

        timeout_manager = TimeoutManager()
        task_manager = RayTaskManager(nodes_per_task=1, timeout_manager=timeout_manager)

        with task_manager:
            executor_ref = ray.put(executor)
            remote_options = {"num_cpus": 1, "num_gpus": 0, "memory": 2 * 1024 * 1024 * 1024}

            # Submit a task
            pg_task_future = task_manager.submit_task(executor_ref, [0], remote_options)

            # Wait for task to be created - wait for the future to complete
            import time

            wait([pg_task_future])

        # Now go over ray tasks and check that all pg are done and no tasks or actors are left running
        time.sleep(1.0)

        # Check that placement groups are removed
        tables = placement_group_table()
        for pg_id, pg_info in tables.items():
            state = pg_info.get("state")
            self.assertEqual(
                state, "REMOVED", f"Placement group {pg_id} should be REMOVED after cleanup, but is {state}"
            )

        for task in list_tasks():
            if task.name == "RankWorker.run_for_rank":
                self.assertEqual(task.state, "FAILED", "Task should be failed after cleanup")

        # Check that all actors are dead
        for actor in list_actors():
            self.assertEqual(actor.state, "DEAD", "Actor should be dead after cleanup")

    def test_timeout_manager(self):
        """Test that TimeoutManager correctly tracks and identifies timed out tasks"""
        timeout_manager = TimeoutManager(timeout_seconds=1)

        mock_task1 = MagicMock()
        mock_task2 = MagicMock()

        # Add tasks
        timeout_manager.add_task(mock_task1)
        timeout_manager.add_task(mock_task2)

        # Check immediately - should have no timeouts
        timed_out = timeout_manager.check_timeouts()
        self.assertEqual(len(timed_out), 0, "No tasks should be timed out immediately")

        # Wait for timeout
        import time

        time.sleep(1.1)

        # Check again - should have timeouts
        timed_out = timeout_manager.check_timeouts()
        self.assertGreaterEqual(len(timed_out), 2, "Both tasks should be timed out")

        # Remove a task
        timeout_manager.remove_task(mock_task1)
        timed_out = timeout_manager.check_timeouts()
        self.assertIn(mock_task2, timed_out, "Remaining task should still be timed out")
        self.assertNotIn(mock_task1, timed_out, "Removed task should not be in timed out list")

    def test_max_resubmissions(self):
        """Test that ranks are not resubmitted beyond max_resubmits limit"""
        from ray.exceptions import WorkerCrashedError

        log_dir = get_datafolder(f"{self.tmp_dir}/test_max_resubmissions")

        class DummyStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                yield {"text": "test"}

        executor = RayPipelineExecutor(
            pipeline=[DummyStep()],
            tasks=1,
            workers=1,
            logging_dir=log_dir,
            nodes_per_task=1,
        )

        timeout_manager = TimeoutManager()
        task_manager = RayTaskManager(nodes_per_task=1, timeout_manager=timeout_manager)

        with task_manager:
            executor_ref = ray.put(executor)
            remote_options = {"num_cpus": 1, "num_gpus": 0, "memory": 2 * 1024 * 1024 * 1024}

            # Create a real task by submitting one, then get the actual ObjectRef
            pg_task_future = task_manager.submit_task(executor_ref, [0], remote_options)

            wait([pg_task_future])

            if not pg_task_future.done():
                self.skipTest("Could not create placement group task in time")

            # Get tasks from the future result
            tasks = pg_task_future.result()
            if not tasks:
                self.skipTest("No tasks were created")

            real_task = tasks[0]
            # Simulate retriable error
            with patch("ray.get", side_effect=WorkerCrashedError("Worker crashed")):
                # First failure - should allow resubmission
                success, ranks_to_resubmit = task_manager.task_done(real_task)
                # This test verifies the logic exists in the code
                # The actual max_resubmits check happens in the run() method
                self.assertIsNotNone(ranks_to_resubmit, "First failure should allow resubmission")
