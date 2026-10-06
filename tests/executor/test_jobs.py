import json
import math
import os
import unittest
from unittest.mock import MagicMock, patch

import dill

from datatrove.executor.jobs import ARRAY_INDEX_ENV_VAR, COORDINATOR_ENV_VAR, JobsPipelineExecutor
from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep


EMITTED_PER_RANK = 5


class _EmitBlock(PipelineStep):
    """Trivial pipeline step so each rank has something to process (mirrors test_ray's SleepBlock)."""

    name = "emit"
    type = "test"

    def run(self, data, rank=None, world_size=None):
        for i in range(EMITTED_PER_RANK):
            self.stat_update("emitted")
            yield i


def _stat_totals(stats_json, name):
    """Collect every numeric total recorded under ``name`` anywhere in a serialized PipelineStats."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == name:
                    found.append(value["total"] if isinstance(value, dict) and "total" in value else value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(stats_json)
    return found


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


def _make_depends_fake():
    """A run_uv_job/inspect_job pair that works across a depends= chain (loads each stage's pik from args)."""
    import fsspec

    launched = []

    def fake_run_uv_job(script, *, script_args=None, env=None, **kwargs):
        with fsspec.open(script_args[0], "rb") as f:  # the pik path (resolve_paths keeps the scheme)
            worker = dill.load(f)
        prev = os.environ.get(ARRAY_INDEX_ENV_VAR)
        os.environ[ARRAY_INDEX_ENV_VAR] = env[ARRAY_INDEX_ENV_VAR]
        try:
            worker.run()
        finally:
            if prev is None:
                os.environ.pop(ARRAY_INDEX_ENV_VAR, None)
            else:
                os.environ[ARRAY_INDEX_ENV_VAR] = prev
        job = MagicMock()
        job.id = f"job-{len(launched)}"
        launched.append(env[ARRAY_INDEX_ENV_VAR])
        return job

    def fake_inspect_job(*, job_id, **kwargs):
        job = MagicMock()
        job.status.stage = "COMPLETED"
        return job

    return fake_run_uv_job, fake_inspect_job, launched


def _make_record_only_fake():
    """A ``run_uv_job`` that records submissions and does no work — the coordinator never runs.

    ``list_jobs`` must be patched alongside it, or the running-coordinator check would hit the Hub.
    """
    submitted: list[dict] = []

    def fake_run_uv_job(script, *, script_args=None, env=None, **kwargs):
        job = MagicMock()
        job.id = f"job-{len(submitted)}"
        job.url = f"https://huggingface.co/jobs/{job.id}"
        submitted.append({"script": script, "script_args": script_args, "env": env, **kwargs})
        return job

    return fake_run_uv_job, submitted


def _make_coordinator_fakes():
    """``run_uv_job`` that plays the part of the Hub, coordinator Job and worker Job alike.

    Each call rehydrates the pickle the way ``launch_pickled_pipeline`` does inside a real Job and runs
    it in a process env holding only the role variable it was submitted with, so the coordinator
    submission recurses into the launcher while a worker submission runs its ranks.
    """
    import fsspec

    submitted: list[dict] = []

    def fake_run_uv_job(script, *, script_args=None, env=None, **kwargs):
        submitted.append(env or {})
        with fsspec.open(script_args[0], "rb") as f:  # the pik path (resolve_paths keeps the scheme)
            job_executor = dill.load(f)
        saved = {key: os.environ.pop(key, None) for key in (ARRAY_INDEX_ENV_VAR, COORDINATOR_ENV_VAR)}
        os.environ.update({k: v for k, v in (env or {}).items() if k in saved})
        try:
            job_executor.run()
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
        job = MagicMock()
        job.id = f"job-{len(submitted)}"
        job.url = f"https://huggingface.co/jobs/{job.id}"
        return job

    def fake_inspect_job(*, job_id, **kwargs):
        job = MagicMock()
        job.status.stage = "COMPLETED"
        return job

    return fake_run_uv_job, fake_inspect_job, submitted


def _fake_list_jobs(*jobs):
    """A ``list_jobs`` that always returns these Jobs (built per iteration, never a late-bound loop var)."""
    return lambda **kwargs: list(jobs)


def _fake_job(labels: dict, stage: str):
    job = MagicMock()
    job.id = "existing-coordinator"
    job.labels = labels
    job.status.stage = stage
    return job


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
                executor.launch_job()

            self.assertEqual(len(launched), math.ceil(tasks / tasks_per_job))
            file_list = ["executor.json", "stats.json"] + [
                x
                for rank in range(tasks)
                for x in (f"completions/{rank:05d}", f"logs/task_{rank:05d}.log", f"stats/{rank:05d}.json")
            ]
            for file in file_list:
                self.assertTrue(log_dir.isfile(file), f"Expected file {file} not found in {log_dir}")

            # per-rank stats must reflect only that rank's work, even when tasks_per_job > 1 packs
            # several ranks into one worker process (guards the per-rank pipeline deepcopy).
            for rank in range(tasks):
                with log_dir.open(f"stats/{rank:05d}.json", "r") as f:
                    emitted = _stat_totals(json.load(f), "emitted")
                self.assertTrue(
                    emitted and all(e == EMITTED_PER_RANK for e in emitted),
                    f"rank {rank} stats leaked across ranks (expected {EMITTED_PER_RANK}, got {emitted})",
                )

    def test_hf_logging_repo_auto_created(self):
        """An hf:// logging repo/bucket that doesn't exist yet is created before the first listing."""
        cases = [
            (
                "hf://datasets/some_user/some-logs/subdir",
                "create_repo",
                "some_user/some-logs",
                {"repo_type": "dataset"},
            ),
            (
                "hf://datasets/some_user/some-logs@refs%2Fconvert%2Fparquet",
                "create_repo",
                "some_user/some-logs",
                {"repo_type": "dataset"},
            ),
            ("hf://spaces/some_user/some-space/logs", "create_repo", "some_user/some-space", {"repo_type": "space"}),
            (
                "hf://some_user/some-model-logs/subdir",
                "create_repo",
                "some_user/some-model-logs",
                {"repo_type": "model"},
            ),
            ("hf://buckets/some_user/some-bucket/logs", "create_bucket", "some_user/some-bucket", {}),
        ]
        for path, expected_fn, expected_id, extra_kwargs in cases:
            executor = JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=2, logging_dir=path, token="hf_faketoken")
            with (
                patch("huggingface_hub.create_repo") as create_repo,
                patch("huggingface_hub.create_bucket") as create_bucket,
            ):
                executor._ensure_logging_dir_repo()
            called = {"create_repo": create_repo, "create_bucket": create_bucket}[expected_fn]
            not_called = create_bucket if expected_fn == "create_repo" else create_repo
            called.assert_called_once_with(
                expected_id, private=True, exist_ok=True, token="hf_faketoken", **extra_kwargs
            )
            not_called.assert_not_called()

    def test_non_hf_logging_dir_not_auto_created(self):
        """Auto-creation is hf://-only: other remote filesystems are left untouched."""
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=2, logging_dir=get_datafolder("memory://jobs-test/no-create"), token="t"
        )
        with (
            patch("huggingface_hub.create_repo") as create_repo,
            patch("huggingface_hub.create_bucket") as create_bucket,
        ):
            executor._ensure_logging_dir_repo()
        create_repo.assert_not_called()
        create_bucket.assert_not_called()

    def test_python_defaults_to_coordinator_version(self):
        """dill ships version-specific bytecode: an unset python= must match the coordinator."""
        import sys

        log_dir = get_datafolder("memory://jobs-test/python-default")
        executor = JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=1, logging_dir=log_dir)
        self.assertEqual(executor.python, f"{sys.version_info.major}.{sys.version_info.minor}")
        explicit = JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=1, logging_dir=log_dir, python="3.11")
        self.assertEqual(explicit.python, "3.11")

    def test_invalid_workers_and_tasks_per_job_rejected(self):
        """Nonsensical concurrency settings fail loudly instead of silently launching nothing."""
        log_dir = get_datafolder("memory://jobs-test/validation")
        for kwargs in ({"tasks_per_job": 0}, {"tasks_per_job": -2}, {"workers": 0}, {"workers": -3}):
            with self.assertRaises(ValueError):
                JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=2, logging_dir=log_dir, **kwargs)

    def test_env_passthrough_to_jobs(self):
        """env= must reach run_uv_job, with user keys overriding defaults but never the array index."""
        log_dir = get_datafolder("memory://jobs-test/env-passthrough")
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()],
            tasks=1,
            logging_dir=log_dir,
            token="hf_faketoken",
            env={"VLLM_ATTENTION_BACKEND": "FLASHINFER", "PYTHONUNBUFFERED": "0"},
            poll_interval=0,
        )
        fake_run, fake_inspect, _ = _make_fake_jobs(executor)
        seen: dict[str, str] = {}

        def recording_run(script, *, env=None, **kwargs):
            seen.update(env)
            return fake_run(script, env=env, **kwargs)

        with patch("huggingface_hub.run_uv_job", recording_run), patch("huggingface_hub.inspect_job", fake_inspect):
            executor.launch_job()

        self.assertEqual(seen["VLLM_ATTENTION_BACKEND"], "FLASHINFER")
        self.assertEqual(seen["PYTHONUNBUFFERED"], "0")  # user value wins over the executor default
        self.assertEqual(seen["HF_HUB_DISABLE_PROGRESS_BARS"], "1")  # untouched default survives
        self.assertEqual(seen[ARRAY_INDEX_ENV_VAR], "0")  # reserved key still set by the executor
        # unlike secrets, env is part of the persisted executor state (documented contract)
        with log_dir.open("executor.json", "r") as f:
            self.assertIn("VLLM_ATTENTION_BACKEND", f.read())

    def test_env_reserved_key_never_clobbers_array_index(self):
        """Even if the reserved key sneaks past init (e.g. attribute mutation), the launch merge wins."""
        log_dir = get_datafolder("memory://jobs-test/env-clobber")
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=1, logging_dir=log_dir, token="hf_faketoken", poll_interval=0
        )
        executor.env = {ARRAY_INDEX_ENV_VAR: "7"}  # bypasses the init check on purpose
        fake_run, fake_inspect, _ = _make_fake_jobs(executor)
        seen: dict[str, str] = {}

        def recording_run(script, *, env=None, **kwargs):
            seen.update(env)
            return fake_run(script, env=env, **kwargs)

        with patch("huggingface_hub.run_uv_job", recording_run), patch("huggingface_hub.inspect_job", fake_inspect):
            executor.launch_job()

        self.assertEqual(seen[ARRAY_INDEX_ENV_VAR], "0")  # executor's value, not the injected "7"

    def test_env_reserved_key_rejected(self):
        """Setting the reserved array-index variable via env= must fail loudly at init."""
        log_dir = get_datafolder("memory://jobs-test/env-reserved")
        with self.assertRaises(ValueError):
            JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=1, logging_dir=log_dir, env={ARRAY_INDEX_ENV_VAR: "7"})

    def test_credentials_not_persisted_to_logging_dir(self):
        """token= and secrets= must never end up in executor.pik / executor.json on the shared logging_dir."""
        log_dir = get_datafolder("memory://jobs-test/credentials")
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()],
            tasks=2,
            logging_dir=log_dir,
            token="hf_faketoken",
            secrets={"MY_SECRET": "sekritvalue"},
            poll_interval=0,
        )
        fake_run, fake_inspect, launched = _make_fake_jobs(executor)
        with patch("huggingface_hub.run_uv_job", fake_run), patch("huggingface_hub.inspect_job", fake_inspect):
            executor.launch_job()

        self.assertEqual(len(launched), 2)  # the scrubbed pickle must still run fine worker-side
        with log_dir.open("executor.pik", "rb") as f:
            pik = f.read()
        with log_dir.open("executor.json", "r") as f:
            as_json = f.read()
        for credential in (b"hf_faketoken", b"sekritvalue"):
            self.assertNotIn(credential, pik)
            self.assertNotIn(credential.decode(), as_json)
        # scrubbing must not clobber the live executor's own credentials
        self.assertEqual(executor.token, "hf_faketoken")
        self.assertEqual(executor.secrets, {"MY_SECRET": "sekritvalue"})

    def test_stats_merged_with_skip_completed_false(self):
        """skip_completed=False disables skipping, but the final stats merge must still happen."""
        log_dir = get_datafolder("memory://jobs-test/no-skip")
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()],
            tasks=2,
            logging_dir=log_dir,
            token="hf_faketoken",
            poll_interval=0,
            skip_completed=False,
        )
        fake_run, fake_inspect, _ = _make_fake_jobs(executor)
        with patch("huggingface_hub.run_uv_job", fake_run), patch("huggingface_hub.inspect_job", fake_inspect):
            executor.launch_job()
        self.assertTrue(log_dir.isfile("stats.json"))

    def test_rerun_merges_stats_if_missing(self):
        """A rerun over a fully-completed dir recreates stats.json if the first coordinator died pre-merge."""
        log_dir = get_datafolder("memory://jobs-test/stats-rerun")

        def build():
            return JobsPipelineExecutor(
                pipeline=[_EmitBlock()], tasks=2, logging_dir=log_dir, token="hf_faketoken", poll_interval=0
            )

        first = build()
        fr, fi, _ = _make_fake_jobs(first)
        with patch("huggingface_hub.run_uv_job", fr), patch("huggingface_hub.inspect_job", fi):
            first.launch_job()
        log_dir.rm("stats.json")  # simulate a coordinator killed after the last rank but before the merge

        second = build()
        fr2, fi2, launched = _make_fake_jobs(second)
        with patch("huggingface_hub.run_uv_job", fr2), patch("huggingface_hub.inspect_job", fi2):
            second.launch_job()
        self.assertEqual(len(launched), 0)
        self.assertTrue(log_dir.isfile("stats.json"))

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
            first.launch_job()
        self.assertEqual(len(launched_first), 4)

        second = build()
        fr2, fi2, launched_second = _make_fake_jobs(second)
        with patch("huggingface_hub.run_uv_job", fr2), patch("huggingface_hub.inspect_job", fi2):
            second.launch_job()
        self.assertEqual(len(launched_second), 0)

    def test_failed_jobs_resubmitted_up_to_max_retries(self):
        """A crashed index is resubmitted and completes on retry; one that keeps crashing is given up on."""
        log_dir = get_datafolder("memory://jobs-test/retries")
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=3, logging_dir=log_dir, token="t", poll_interval=0, max_retries=1
        )
        attempts: dict[str, int] = {}
        job_stages: dict[str, str] = {}

        def fake_run(script, *, script_args=None, env=None, **kwargs):
            idx = env[ARRAY_INDEX_ENV_VAR]
            attempts[idx] = attempts.get(idx, 0) + 1
            job = MagicMock()
            job.id = f"job-{idx}-attempt{attempts[idx]}"
            # index 1 crashes on its first attempt only; index 2 crashes every time
            if (idx == "1" and attempts[idx] == 1) or idx == "2":
                job_stages[job.id] = "ERROR"  # died before running any rank
                return job
            with executor.logging_dir.open("executor.pik", "rb") as f:
                worker = dill.load(f)
            prev = os.environ.get(ARRAY_INDEX_ENV_VAR)
            os.environ[ARRAY_INDEX_ENV_VAR] = idx
            try:
                worker.run()
            finally:
                if prev is None:
                    os.environ.pop(ARRAY_INDEX_ENV_VAR, None)
                else:
                    os.environ[ARRAY_INDEX_ENV_VAR] = prev
            job_stages[job.id] = "COMPLETED"
            return job

        def fake_inspect(*, job_id, **kwargs):
            job = MagicMock()
            job.status.stage = job_stages[job_id]
            return job

        with patch("huggingface_hub.run_uv_job", fake_run), patch("huggingface_hub.inspect_job", fake_inspect):
            executor.launch_job()

        self.assertEqual(attempts, {"0": 1, "1": 2, "2": 2})  # one resubmit each for 1 and 2, then give up on 2
        self.assertTrue(log_dir.isfile("completions/00000"))
        self.assertTrue(log_dir.isfile("completions/00001"))  # completed on the resubmit
        self.assertFalse(log_dir.isfile("completions/00002"))  # retries exhausted -> left incomplete
        self.assertFalse(log_dir.isfile("stats.json"))  # an incomplete run must not merge stats

    def test_failed_rerun_does_not_merge_stale_stats(self):
        """A skip_completed=False rerun whose ranks all fail must not rebuild stats.json from stale markers."""
        log_dir = get_datafolder("memory://jobs-test/stale-stats")

        first = JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=2, logging_dir=log_dir, token="t", poll_interval=0)
        fr, fi, _ = _make_fake_jobs(first)
        with patch("huggingface_hub.run_uv_job", fr), patch("huggingface_hub.inspect_job", fi):
            first.launch_job()
        self.assertTrue(log_dir.isfile("stats.json"))
        log_dir.rm("stats.json")

        second = JobsPipelineExecutor(
            pipeline=[_EmitBlock()],
            tasks=2,
            logging_dir=log_dir,
            token="t",
            poll_interval=0,
            skip_completed=False,
            max_retries=0,
        )

        def failing_run(script, *, script_args=None, env=None, **kwargs):
            job = MagicMock()  # dies before running any rank
            job.id = f"failing-{env[ARRAY_INDEX_ENV_VAR]}"
            return job

        def error_inspect(*, job_id, **kwargs):
            job = MagicMock()
            job.status.stage = "ERROR"
            return job

        with patch("huggingface_hub.run_uv_job", failing_run), patch("huggingface_hub.inspect_job", error_inspect):
            second.launch_job()
        # run 1's completion markers would mask run 2's failures — the merge must not run at all
        self.assertFalse(log_dir.isfile("stats.json"))

    def test_depends_runs_parent_then_child(self):
        """`depends=` launches the parent to completion, then runs the child; both write full layouts."""
        parent_dir = get_datafolder("memory://jobs-test/dep-parent")
        child_dir = get_datafolder("memory://jobs-test/dep-child")
        parent = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=2, logging_dir=parent_dir, token="t", poll_interval=0
        )
        child = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=2, logging_dir=child_dir, token="t", poll_interval=0, depends=parent
        )
        fr, fi, launched = _make_depends_fake()
        with patch("huggingface_hub.run_uv_job", fr), patch("huggingface_hub.inspect_job", fi):
            child.launch_job()
        self.assertEqual(len(launched), 4)  # 2 parent + 2 child array indices
        for log_dir in (parent_dir, child_dir):
            self.assertEqual(len(log_dir.list_files("completions")), 2)

    def test_depends_fails_fast_when_dependency_incomplete(self):
        """If the dependency finishes with failed tasks, the child raises instead of waiting forever."""
        parent_dir = get_datafolder("memory://jobs-test/dep-fail-parent")
        child_dir = get_datafolder("memory://jobs-test/dep-fail-child")
        parent = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=2, logging_dir=parent_dir, token="t", poll_interval=0, max_retries=0
        )
        child = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=2, logging_dir=child_dir, token="t", poll_interval=0, depends=parent
        )

        def failing_run(script, *, script_args=None, env=None, **kwargs):
            job = MagicMock()  # never runs the ranks -> no completions written
            job.id = "failing"
            return job

        def error_inspect(*, job_id, **kwargs):
            job = MagicMock()
            job.status.stage = "ERROR"
            return job

        with patch("huggingface_hub.run_uv_job", failing_run), patch("huggingface_hub.inspect_job", error_inspect):
            with self.assertRaises(RuntimeError):
                child.launch_job()

    def test_run_submits_one_coordinator_job_and_returns(self):
        """`run()` submits one coordinator Job and returns without running anything in this process."""
        log_dir = get_datafolder("memory://jobs-coordinator/submits")
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()],
            tasks=3,
            workers=2,
            logging_dir=log_dir,
            token="hf_faketoken",
            secrets={"MY_SECRET": "sekritvalue"},
            env={"MY_SETTING": "my_value"},
            coordinator_flavor="cpu-upgrade",
            coordinator_timeout="9d",
            poll_interval=0,
        )
        fake_run, submitted = _make_record_only_fake()
        with patch("huggingface_hub.run_uv_job", fake_run), patch("huggingface_hub.list_jobs", lambda **kw: []):
            executor.run()

        self.assertEqual(len(submitted), 1)
        coordinator = submitted[0]
        self.assertEqual(coordinator["flavor"], "cpu-upgrade")
        self.assertEqual(coordinator["timeout"], "9d")
        self.assertEqual(coordinator["labels"]["datatrove_role"], "coordinator")
        self.assertEqual(coordinator["labels"]["datatrove_job_name"], executor.job_name)
        # the coordinator must not look like a worker: no rank addressing, and its own role marker
        self.assertNotIn(ARRAY_INDEX_ENV_VAR, coordinator["env"])
        self.assertEqual(coordinator["env"][COORDINATOR_ENV_VAR], "1")
        # ... but it needs the env/secrets the Jobs get, to submit them and to reach the logging_dir
        self.assertEqual(coordinator["env"]["MY_SETTING"], "my_value")
        self.assertEqual(coordinator["secrets"]["HF_TOKEN"], "hf_faketoken")
        self.assertEqual(coordinator["secrets"]["MY_SECRET"], "sekritvalue")
        self.assertEqual(coordinator["script_args"], [log_dir.resolve_paths("executor.pik")])

        # it must be able to rehydrate the executor, and the pickle must hold no credentials
        with log_dir.open("executor.pik", "rb") as f:
            pickled = dill.load(f)
        self.assertIsNone(pickled.token)
        self.assertIsNone(pickled.secrets)

        # fire-and-forget: nothing was launched or merged locally, not even ranks_to_run.json
        self.assertFalse(log_dir.isfile("stats.json"))
        self.assertFalse(log_dir.isfile("ranks_to_run.json"))
        self.assertFalse(any(log_dir.isfile(f"completions/{rank:05d}") for rank in range(3)))

    def test_launch_job_supervises_locally_without_a_coordinator_job(self):
        """`launch_job()` is how to opt out: the pool runs in this process, and no Job is a coordinator.

        The fake worker demands ``DATATROVE_JOB_ARRAY_INDEX``, so it would raise on any coordinator Job.
        """
        log_dir = get_datafolder("memory://jobs-coordinator/local-supervisor")
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=2, logging_dir=log_dir, token="hf_faketoken", poll_interval=0
        )
        fake_run, fake_inspect, launched = _make_fake_jobs(executor)
        with patch("huggingface_hub.run_uv_job", fake_run), patch("huggingface_hub.inspect_job", fake_inspect):
            executor.launch_job()

        self.assertEqual(len(launched), 2)
        self.assertEqual(len(log_dir.list_files("completions")), 2)
        self.assertTrue(log_dir.isfile("stats.json"))

    def test_coordinator_job_supervises_the_pool(self):
        """The coordinator Job — not the local process — launches the workers and merges the stats."""
        log_dir = get_datafolder("memory://jobs-coordinator/inside")
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()],
            tasks=3,
            tasks_per_job=2,
            logging_dir=log_dir,
            token="hf_faketoken",
            poll_interval=0,
        )
        fake_run, fake_inspect, submitted = _make_coordinator_fakes()
        with (
            patch("huggingface_hub.run_uv_job", fake_run),
            patch("huggingface_hub.inspect_job", fake_inspect),
            patch("huggingface_hub.list_jobs", lambda **kw: []),
        ):
            executor.run()

        self.assertEqual(len(submitted), 1 + math.ceil(3 / 2))  # one coordinator, then 2 worker Jobs
        # exactly one coordinator generation: inside the Job the executor launches, it does not re-spawn
        self.assertEqual(sum(COORDINATOR_ENV_VAR in env for env in submitted), 1)
        self.assertEqual(sum(ARRAY_INDEX_ENV_VAR in env for env in submitted), 2)
        self.assertEqual(len(log_dir.list_files("completions")), 3)
        self.assertTrue(log_dir.isfile("stats.json"))  # merged by the coordinator
        self.assertIsNone(os.environ.get(COORDINATOR_ENV_VAR))  # the fake left this process alone
        self.assertTrue(executor._launched)

    def test_run_does_not_launch_a_second_coordinator(self):
        """A live coordinator for the same run makes a relaunch a no-op; a finished one doesn't."""
        log_dir = get_datafolder("memory://jobs-coordinator/twice")

        def build():
            return JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=2, logging_dir=log_dir, token="t")

        for stage, expected_submissions in (("RUNNING", 0), ("COMPLETED", 1)):
            executor = build()
            live = _fake_job(executor._coordinator_labels, stage)
            fake_run, submitted = _make_record_only_fake()
            with (
                patch("huggingface_hub.run_uv_job", fake_run),
                patch("huggingface_hub.list_jobs", _fake_list_jobs(live)),
            ):
                executor.run()
            self.assertEqual(len(submitted), expected_submissions, f"{stage} coordinator must not be doubled")

        # a coordinator under another logging_dir is another run, so it must not block this one
        executor = build()
        other = _fake_job({**executor._coordinator_labels, "datatrove_logging_dir": "0" * 12}, "RUNNING")
        fake_run, submitted = _make_record_only_fake()
        with (
            patch("huggingface_hub.run_uv_job", fake_run),
            patch("huggingface_hub.list_jobs", _fake_list_jobs(other)),
        ):
            executor.run()
        self.assertEqual(len(submitted), 1)

    def test_dependent_run_submits_the_whole_chain_but_a_small_pickle(self):
        """Calling `run()` on the last stage alone still starts every stage, each waiting on the previous."""
        parent_dir = get_datafolder("memory://jobs-coordinator/chain-parent")
        child_dir = get_datafolder("memory://jobs-coordinator/chain-child")
        parent = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=2, logging_dir=parent_dir, token="t", job_name="parent"
        )
        child = JobsPipelineExecutor(
            pipeline=[_EmitBlock()],
            tasks=2,
            logging_dir=child_dir,
            token="t",
            job_name="child",
            depends=parent,
        )
        fake_run, submitted = _make_record_only_fake()
        with patch("huggingface_hub.run_uv_job", fake_run), patch("huggingface_hub.list_jobs", lambda **kw: []):
            child.run()

        # the dependency has no launcher of its own to wait on, so run() submits it too, upstream first
        self.assertEqual([job["labels"]["datatrove_job_name"] for job in submitted], ["parent", "child"])
        self.assertFalse(parent_dir.isfile("ranks_to_run.json"))  # submitted, but nothing runs from here
        self.assertFalse(child_dir.isfile("ranks_to_run.json"))

        with child_dir.open("executor.pik", "rb") as f:
            pickled = dill.load(f)
        self.assertIsNotNone(pickled.depends)
        self.assertTrue(pickled.depends._launched)  # -> launch_job() waits for completions/ only
        self.assertEqual(pickled.depends.depends, None)
        self.assertEqual(pickled.depends.pipeline, [])  # the parent's pipeline stays out of the pickle
        self.assertEqual(len(parent.pipeline), 1)  # the live parent object was not mutated

    def test_env_coordinator_key_rejected(self):
        """The coordinator role variable is reserved, like the array-index one."""
        log_dir = get_datafolder("memory://jobs-coordinator/env-reserved")
        with self.assertRaises(ValueError):
            JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=1, logging_dir=log_dir, env={COORDINATOR_ENV_VAR: "1"})


if __name__ == "__main__":
    unittest.main()
