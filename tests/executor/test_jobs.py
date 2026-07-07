import math
import os
import unittest
from unittest.mock import MagicMock, patch

import dill

from datatrove.executor.jobs import ARRAY_INDEX_ENV_VAR, JobsPipelineExecutor
from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep


class _EmitBlock(PipelineStep):
    """Trivial pipeline step so each rank has something to process (mirrors test_ray's SleepBlock)."""

    name = "emit"
    type = "test"

    def run(self, data, rank=None, world_size=None):
        for i in range(5):
            yield i


def _make_fake_jobs(executor: JobsPipelineExecutor):
    """Build patched ``run_uv_job`` / ``inspect_job`` that run each array index in-process.

    ``run_uv_job`` is patched to do exactly what a real Job does: set ``DATATROVE_JOB_ARRAY_INDEX``,
    rehydrate the pickled executor from the shared ``logging_dir`` (exercising the real dill round-trip),
    and call ``run()`` so the worker path executes its block of ranks. ``inspect_job`` always reports
    ``COMPLETED`` since the work already ran synchronously.
    """
    launched: dict[str, str] = {}

    def fake_run_uv_job(script, *, script_args=None, env=None, **kwargs):
        array_index = env[ARRAY_INDEX_ENV_VAR]
        with executor.logging_dir.open("executor.pik", "rb") as f:
            worker = dill.load(f)
        prev = os.environ.get(ARRAY_INDEX_ENV_VAR)
        os.environ[ARRAY_INDEX_ENV_VAR] = array_index
        try:
            worker.run()
        finally:
            if prev is None:
                os.environ.pop(ARRAY_INDEX_ENV_VAR, None)
            else:
                os.environ[ARRAY_INDEX_ENV_VAR] = prev
        job = MagicMock()
        job.id = f"job-{array_index}"
        job.url = f"https://hf.co/jobs/job-{array_index}"
        launched[job.id] = array_index
        return job

    def fake_inspect_job(*, job_id, **kwargs):
        job = MagicMock()
        job.status.stage = "COMPLETED"
        return job

    return fake_run_uv_job, fake_inspect_job, launched


class TestJobsExecutor(unittest.TestCase):
    def test_local_logging_dir_rejected(self):
        """A local logging_dir cannot be shared with remote Jobs, so it must raise."""
        with self.assertRaises(ValueError):
            JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=2, logging_dir="/tmp/datatrove-jobs-local")

    def test_fan_out_writes_full_layout(self):
        """Fanning out over N array indices writes the standard datatrove file layout for every rank."""
        for tasks, tasks_per_job, workers in ((6, 2, 2), (3, 1, -1)):
            log_dir = get_datafolder(f"memory://jobs-test/{tasks}-{tasks_per_job}-{workers}")
            executor = JobsPipelineExecutor(
                pipeline=[_EmitBlock()],
                tasks=tasks,
                tasks_per_job=tasks_per_job,
                workers=workers,
                logging_dir=log_dir,
                token="hf_faketoken",
                poll_interval=0,
            )
            fake_run, fake_inspect, launched = _make_fake_jobs(executor)
            with (
                patch("huggingface_hub.run_uv_job", fake_run),
                patch("huggingface_hub.inspect_job", fake_inspect),
            ):
                executor.run()

            self.assertEqual(len(launched), math.ceil(tasks / tasks_per_job))
            file_list = ["executor.json", "stats.json"] + [
                x
                for rank in range(tasks)
                for x in (f"completions/{rank:05d}", f"logs/task_{rank:05d}.log", f"stats/{rank:05d}.json")
            ]
            for file in file_list:
                self.assertTrue(log_dir.isfile(file), f"Expected file {file} not found in {log_dir}")

    def test_resume_skips_completed_ranks(self):
        """A second run over an already-completed logging_dir launches no Jobs (idempotent resume)."""
        log_dir = get_datafolder("memory://jobs-test/resume")

        def build():
            return JobsPipelineExecutor(
                pipeline=[_EmitBlock()],
                tasks=4,
                tasks_per_job=1,
                logging_dir=log_dir,
                token="hf_faketoken",
                poll_interval=0,
            )

        first = build()
        fr, fi, launched_first = _make_fake_jobs(first)
        with patch("huggingface_hub.run_uv_job", fr), patch("huggingface_hub.inspect_job", fi):
            first.run()
        self.assertEqual(len(launched_first), 4)

        second = build()
        fr2, fi2, launched_second = _make_fake_jobs(second)
        with patch("huggingface_hub.run_uv_job", fr2), patch("huggingface_hub.inspect_job", fi2):
            second.run()
        self.assertEqual(len(launched_second), 0)


if __name__ == "__main__":
    unittest.main()
