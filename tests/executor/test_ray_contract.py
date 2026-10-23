"""Characterization tests pinning the RayPipelineExecutor per-rank contract.

These tests run `_run_for_rank` directly (no Ray cluster required) and pin the
behavior that the rest of the datatrove ecosystem depends on:

- the DATATROVE_* environment variables (see `pipeline/inference/distributed/utils.py`)
- master-only writes of stats and completion markers (`node_rank <= 0`)
- the on-disk layout inside `logging_dir`
- skip-completed (resumability) semantics

They exist so the executor internals can be reworked without breaking consumers.
"""

import json
import os
import shutil
import tempfile
import unittest

from datatrove.executor.ray import RayPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.stats import PipelineStats


ENV_KEYS = [
    "DATATROVE_NODE_RANK",
    "DATATROVE_EXECUTOR",
    "DATATROVE_NODE_IPS",
    "DATATROVE_CPUS_PER_TASK",
    "DATATROVE_MEM_PER_CPU",
    "DATATROVE_GPUS_ON_NODE",
]


class CountingStep(PipelineStep):
    """Yields `num_docs` documents and counts how many times it ran."""

    name = "CountingStep"
    type = "Test"

    def __init__(self, num_docs: int = 3):
        super().__init__()
        self.num_docs = num_docs
        self.times_called = 0

    def run(self, data, rank: int, world_size: int):
        self.times_called += 1
        for i in range(self.num_docs):
            self.stat_update("docs", value=1)
            yield {"text": f"doc {i} from rank {rank}", "id": f"{rank}-{i}"}


class TestRayExecutorEnvContract(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        # _set_distributed_environment mutates os.environ; snapshot and restore
        self._saved_env = {k: os.environ.get(k) for k in ENV_KEYS + ["RAY_NODELIST"]}
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _make_executor(self, tasks: int = 2, **kwargs) -> tuple[RayPipelineExecutor, CountingStep]:
        step = CountingStep()
        executor = RayPipelineExecutor(
            pipeline=[step],
            tasks=tasks,
            logging_dir=f"{self.tmp_dir}/logs_{id(step)}",
            **kwargs,
        )
        return executor, step

    def test_env_vars_single_node(self):
        """node_rank=-1 (default) must produce the single-node contract values."""
        executor, _ = self._make_executor()
        os.environ.pop("RAY_NODELIST", None)

        executor._run_for_rank(rank=0)

        self.assertEqual(os.environ["DATATROVE_EXECUTOR"], "RAY")
        self.assertEqual(os.environ["DATATROVE_NODE_RANK"], "-1")
        self.assertEqual(os.environ["DATATROVE_NODE_IPS"], "")
        self.assertEqual(os.environ["DATATROVE_CPUS_PER_TASK"], "1")
        self.assertEqual(os.environ["DATATROVE_MEM_PER_CPU"], "2")
        self.assertEqual(os.environ["DATATROVE_GPUS_ON_NODE"], "0")

    def test_env_vars_multinode(self):
        """Each gang member must see the full, ordered node list and its own node_rank.

        In real runs RAY_NODELIST is set by the executor's worker entry points (native
        task or gang actor) before calling _run_for_rank; here we set it manually to
        simulate the worker-side environment.
        """
        executor, _ = self._make_executor()
        os.environ["RAY_NODELIST"] = "10.0.0.1,10.0.0.2,10.0.0.3"

        executor._run_for_rank(rank=0, node_rank=2)

        self.assertEqual(os.environ["DATATROVE_NODE_RANK"], "2")
        self.assertEqual(os.environ["DATATROVE_NODE_IPS"], "10.0.0.1,10.0.0.2,10.0.0.3")
        # value must be a string: consumers do int(os.environ[...])
        self.assertIsInstance(os.environ["DATATROVE_NODE_RANK"], str)

    def test_fs_layout_master_writes(self):
        """Master node (node_rank=-1) writes stats + completion marker + task log."""
        executor, _ = self._make_executor(tasks=2)
        log_dir = executor.logging_dir

        executor._run_for_rank(rank=0)

        self.assertTrue(log_dir.isfile("completions/00000"))
        self.assertTrue(log_dir.isfile("stats/00000.json"))
        self.assertTrue(log_dir.isfile("logs/task_00000.log"))
        # no other ranks were run
        self.assertFalse(log_dir.isfile("completions/00001"))

    def test_fs_layout_worker_does_not_write(self):
        """Worker nodes (node_rank > 0) must NOT write stats or completion markers.

        This is what prevents N gang nodes from writing the same rank's output N times.
        """
        executor, _ = self._make_executor()
        log_dir = executor.logging_dir

        executor._run_for_rank(rank=0, node_rank=1)

        self.assertFalse(log_dir.isfile("completions/00000"))
        self.assertFalse(log_dir.isfile("stats/00000.json"))
        # the task log is still uploaded for post-mortem debugging
        self.assertTrue(log_dir.isfile("logs/task_00000.log"))

    def test_stats_file_round_trip(self):
        """stats/{rank:05d}.json must parse back into a PipelineStats with the step's metrics."""
        executor, _ = self._make_executor()
        stats = executor._run_for_rank(rank=0)

        with executor.logging_dir.open("stats/00000.json", "r") as f:
            parsed = PipelineStats.from_json(json.load(f))
        self.assertEqual(len(parsed.stats), len(stats.stats))
        # the counter stat written by the step must survive serialization (single-value stats round-trip as ints)
        docs = [s for s in parsed.stats if s.name == "Test: CountingStep"]
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].stats["docs"], 3)

    def test_skip_completed(self):
        """A completed rank must not re-run the pipeline (resumability)."""
        executor, step = self._make_executor()

        executor._run_for_rank(rank=0)
        stats = executor._run_for_rank(rank=0)

        self.assertEqual(step.times_called, 1)
        # early return yields empty stats
        self.assertEqual(len(stats.stats), 0)

    def test_no_skip_when_disabled(self):
        """skip_completed=False must re-run completed ranks."""
        executor, step = self._make_executor(skip_completed=False)

        executor._run_for_rank(rank=0)
        executor._run_for_rank(rank=0)

        self.assertEqual(step.times_called, 2)

    def test_world_size_is_tasks(self):
        executor, _ = self._make_executor(tasks=7)
        self.assertEqual(executor.world_size, 7)

    def test_workers_default_unlimited(self):
        executor, _ = self._make_executor(tasks=7)
        self.assertEqual(executor.workers, 7)


if __name__ == "__main__":
    unittest.main()
