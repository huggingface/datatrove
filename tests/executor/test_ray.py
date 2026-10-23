#!/usr/bin/env python3
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import ray
from ray.util.placement_group import placement_group_table
from ray.util.state import list_actors, list_tasks

from datatrove.executor.ray import PG_READY_TIMEOUT_S, RankWorker, RayPipelineExecutor, run_rank_task
from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep


RUNNING_STATES = ("PENDING_ARG", "PENDING_NODE_ASSIGNMENT", "SUBMITTED_TO_WORKER", "RECEIVED_BY_WORKER", "RUNNING")


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

    def test_placement_group_creation_bounded_failure(self):
        """On a 1-node cluster, nodes_per_task=2 with STRICT_SPREAD can never be satisfied: the run must fail
        bounded and informatively (no hang), leaving no placement group or actor behind."""
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

        with patch("datatrove.executor.ray.PG_READY_TIMEOUT_S", 3):
            with self.assertRaisesRegex(RuntimeError, "STRICT_SPREAD"):
                executor.run()

        self.assertEqual(len(list_actors()), 0, "No gang actor should have been created")
        for info in placement_group_table().values():
            self.assertEqual(info["state"], "REMOVED", "The unsatisfiable placement group must be removed")

    def test_unschedulable_resources_fail_before_submit(self):
        """Per-task resources the cluster cannot satisfy would leave rank tasks PENDING forever: the run must fail
        pre-submit with an informative error naming the job, and no rank task may be submitted."""
        log_dir = get_datafolder(f"{self.tmp_dir}/resource_guard")

        class DummyStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                yield {"text": "test"}

        executor = RayPipelineExecutor(
            pipeline=[DummyStep()], tasks=2, workers=2, logging_dir=log_dir, cpus_per_task=4
        )

        with patch("ray.available_resources", return_value={"CPU": 2, "GPU": 0.0, "memory": 4 * 1024**3}):
            with self.assertRaisesRegex(
                RuntimeError, r"ranks \[0\].*refusing to submit work that can never be scheduled"
            ):
                executor.run()

        self.assertEqual([t for t in list_tasks() if t.name == "run_rank_task"], [], "no rank task may be submitted")
        self.assertFalse(log_dir.isfile("completions/00000"), "no rank may have run")

    def test_in_flight_reservations_keep_queued_jobs(self):
        """This executor's own in-flight tasks drain ray.available_resources() to 0; queued jobs must still be submitted
        (the pre-submit check credits the executor's outstanding reservations instead of destroying legitimate work)."""
        log_dir = get_datafolder(f"{self.tmp_dir}/in_flight_credit")

        class DummyStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                yield {"text": "test"}

        checks = {"n": 0}

        def draining_available_resources():
            # first pre-submit check sees 2 free CPUs; afterwards our own submitted tasks have claimed everything
            checks["n"] += 1
            if checks["n"] == 1:
                return {"CPU": 2, "GPU": 0.0, "memory": 4 * 1024**3}
            return {"CPU": 0, "GPU": 0.0, "memory": 0}

        executor = RayPipelineExecutor(pipeline=[DummyStep()], tasks=3, workers=-1, logging_dir=log_dir)

        # with 2 identical tasks in flight, the third identical job must not fail pre-submit
        with patch("ray.available_resources", side_effect=draining_available_resources):
            executor.run()

        self.assertGreaterEqual(checks["n"], 3, "all three jobs must have gone through the pre-submit check")
        for rank in range(3):
            self.assertTrue(log_dir.isfile(f"completions/{rank:05d}"), f"rank {rank} must have been submitted and run")

    def test_single_node_fs_layout_and_stats_merge(self):
        """All ranks complete and the logging_dir layout (completions/logs/stats + merged stats.json) is correct."""
        log_dir = get_datafolder(f"{self.tmp_dir}/layout")

        class DummyStep(PipelineStep):
            name = "DummyStep"
            type = "Test"

            def run(self, data, rank=None, world_size=None):
                for i in range(2):
                    yield {"text": f"{rank}-{i}"}

        executor = RayPipelineExecutor(pipeline=[DummyStep()], tasks=2, workers=2, logging_dir=log_dir)
        executor.run()

        for rank in range(2):
            self.assertTrue(log_dir.isfile(f"completions/{rank:05d}"))
            self.assertTrue(log_dir.isfile(f"logs/task_{rank:05d}.log"))
            self.assertTrue(log_dir.isfile(f"stats/{rank:05d}.json"))
        self.assertTrue(log_dir.isfile("stats.json"))
        with log_dir.open("stats.json", "r") as f:
            merged = json.load(f)
        self.assertEqual(len(merged), 1)  # one stat: the DummyStep, aggregated over both ranks
        self.assertIn("DummyStep", merged[0]["name"])

    def test_workers_one_runs_sequentially(self):
        """workers=1 must run ranks one at a time, in submission order."""
        log_dir = get_datafolder(f"{self.tmp_dir}/sequential")
        marker = f"{self.tmp_dir}/marker.txt"

        class MarkerStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                with open(marker, "a") as f:
                    f.write(f"start {rank}\n")
                time.sleep(1.0)
                with open(marker, "a") as f:
                    f.write(f"end {rank}\n")
                yield {"text": "test"}

        executor = RayPipelineExecutor(pipeline=[MarkerStep()], tasks=2, workers=1, logging_dir=log_dir)
        executor.run()

        with open(marker, "r") as f:
            self.assertEqual(f.read().splitlines(), ["start 0", "end 0", "start 1", "end 1"])

    def test_application_error_raises_but_siblings_complete(self):
        """An application error must surface as a RuntimeError, while sibling ranks still complete (fail-slow)."""
        log_dir = get_datafolder(f"{self.tmp_dir}/app_error")

        class FlakyStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                if rank == 1:
                    raise RuntimeError("boom")
                yield {"text": "test"}

        executor = RayPipelineExecutor(pipeline=[FlakyStep()], tasks=3, workers=3, logging_dir=log_dir)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            executor.run()

        self.assertTrue(log_dir.isfile("completions/00000"))
        self.assertTrue(log_dir.isfile("completions/00002"))
        self.assertFalse(log_dir.isfile("completions/00001"))

    def test_timeout_cancels_bounded(self):
        """A rank exceeding `time` is cancelled and retried within budget, then fails the run without a marker."""
        log_dir = get_datafolder(f"{self.tmp_dir}/timeout")

        class SlowStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                time.sleep(10)
                yield {"text": "test"}

        executor = RayPipelineExecutor(pipeline=[SlowStep()], tasks=1, workers=1, logging_dir=log_dir, time=2)

        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "time limit"):
            executor.run()
        self.assertLess(time.monotonic() - start, PG_READY_TIMEOUT_S)

        self.assertFalse(log_dir.isfile("completions/00000"), "A timed-out rank must not count as done")

    def test_keyboard_interrupt_propagates_and_cancels(self):
        """KeyboardInterrupt on the driver propagates after cleanup and leaves no task running."""
        log_dir = get_datafolder(f"{self.tmp_dir}/abort")

        class SlowStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                time.sleep(30)
                yield {"text": "test"}

        executor = RayPipelineExecutor(pipeline=[SlowStep()], tasks=2, workers=2, logging_dir=log_dir)

        real_wait = ray.wait
        calls = {"n": 0}

        def interrupt_first_wait(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise KeyboardInterrupt
            return real_wait(*args, **kwargs)

        with patch("ray.wait", new=interrupt_first_wait):
            with self.assertRaises(KeyboardInterrupt):
                executor.run()

        # nothing may still be running after the cleanup
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            states = [t.state for t in list_tasks() if t.name == "run_rank_task"]
            if states and all(state not in RUNNING_STATES for state in states):
                break
            time.sleep(0.5)
        for task in list_tasks():
            if task.name == "run_rank_task":
                self.assertNotIn(task.state, RUNNING_STATES, f"Task {task.task_id} was not cancelled")

    def test_no_placement_groups_or_actors_single_node(self):
        """nodes_per_task=1 must use only native Ray tasks: no placement groups, no actors."""
        log_dir = get_datafolder(f"{self.tmp_dir}/native")

        class DummyStep(PipelineStep):
            def run(self, data, rank=None, world_size=None):
                yield {"text": "test"}

        executor = RayPipelineExecutor(pipeline=[DummyStep()], tasks=2, workers=2, logging_dir=log_dir)
        executor.run()

        self.assertEqual(placement_group_table(), {})
        self.assertEqual(len(list_actors()), 0)
        # ray's state API is eventually consistent: give it a moment to register the finished tasks
        names = set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and names != {"run_rank_task"}:
            names = {t.name for t in list_tasks()}
            time.sleep(0.5)
        self.assertEqual(names, {"run_rank_task"})

    def test_log_first_local_rank_mapping(self):
        """log_first=True keeps rank 0 on console (0-based); log_first=False shifts local ranks to 1-based."""
        executor = RayPipelineExecutor(pipeline=[], tasks=2, logging_dir=get_datafolder(f"{self.tmp_dir}/logfirst"))
        for log_first, first_local_rank in ((False, 1), (True, 0)):
            executor.log_first = log_first
            with patch.object(executor, "_run_for_rank") as run_mock, patch.dict(os.environ):
                run_rank_task(executor, 0, 0)
                RankWorker().run_for_rank(executor, [0, 1], ["1.2.3.4"], 0)
            self.assertEqual(
                [call.args[1] for call in run_mock.call_args_list],
                [first_local_rank, first_local_rank, first_local_rank + 1],
            )


if __name__ == "__main__":
    unittest.main()
