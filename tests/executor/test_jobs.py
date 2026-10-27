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

    def fake_run_uv_job(script, *, script_args=None, env=None, secrets=None, **kwargs):
        submitted.append(env or {})
        with fsspec.open(script_args[0], "rb") as f:  # the pik path (resolve_paths keeps the scheme)
            job_executor = dill.load(f)
        # a real Job sees its role variable and its secrets as env vars, and nothing from the launcher
        keys = {ARRAY_INDEX_ENV_VAR, COORDINATOR_ENV_VAR, "HF_TOKEN", *(secrets or {})}
        saved = {key: os.environ.pop(key, None) for key in keys}
        os.environ.update({k: v for k, v in (env or {}).items() if k in (ARRAY_INDEX_ENV_VAR, COORDINATOR_ENV_VAR)})
        os.environ.update(secrets or {})
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


def _fake_list_jobs(*jobs, record: list | None = None):
    """A ``list_jobs`` that returns these Jobs, keeping those carrying all the labels asked for.

    Mirrors how the Hub filters server-side (built per iteration, never a late-bound loop var). ``record``
    collects the kwargs of every call, so a test can check what was asked for.
    """

    def fake(labels=None, **kwargs):
        if record is not None:
            record.append({"labels": labels, **kwargs})
        return [job for job in jobs if not labels or all((job.labels or {}).get(k) == v for k, v in labels.items())]

    return fake


def _fake_job(labels: dict, stage: str):
    job = MagicMock()
    job.id = "existing-coordinator"
    job.labels = labels
    job.status.stage = stage
    return job


class _ProbeBlock(PipelineStep):
    """Records, per rank, which secret it saw and when it ran; can be told to fail."""

    name = "probe"
    type = "test"

    def __init__(self, stage: str, fail: bool = False):
        super().__init__()
        self.stage = stage
        self.fail = fail

    def run(self, data, rank=None, world_size=None):
        import fsspec

        if self.fail:
            raise RuntimeError(f"stage {self.stage} failed")
        fs = fsspec.filesystem("memory")
        order = len(fs.glob("/jobs-chain-probe/*.json"))  # the fake Hub runs Jobs one at a time
        seen = {"secret": os.environ.get(f"SECRET_{self.stage}"), "order": order}
        with fsspec.open(f"memory://jobs-chain-probe/{self.stage}-{rank}.json", "w") as f:
            json.dump(seen, f)
        yield 1


def _probes():
    import fsspec

    fs = fsspec.filesystem("memory")
    found = {}
    for path in fs.glob("/jobs-chain-probe/*.json"):
        with fs.open(path, "r") as f:
            found[path.rsplit("/", 1)[-1].removesuffix(".json")] = json.load(f)
    return found


def _make_chain_hub():
    """A fake Hub for ``depends=`` chains: runs each Job in-process and records how it ended.

    Like a real Job, each one sees only its role variable and its secrets in its environment, and the
    pickle is read the moment the Job is submitted.
    """
    import fsspec

    jobs: dict[str, dict] = {}
    roles = (ARRAY_INDEX_ENV_VAR, COORDINATOR_ENV_VAR)
    secret_names = {"HF_TOKEN"}  # every secret any Job got: a Job must not see another Job's

    def fake_run_uv_job(script, *, script_args=None, env=None, secrets=None, labels=None, dependencies=None, **kwargs):
        job_id = f"job-{len(jobs)}"
        with fsspec.open(script_args[0], "rb") as f:
            raw = f.read()
        record = {
            "role": "coordinator" if COORDINATOR_ENV_VAR in env else "worker",
            "labels": labels,
            "pik": raw,
            "dependencies": dependencies,
        }
        jobs[job_id] = record
        secret_names.update(secrets or {})
        saved = {key: os.environ.pop(key, None) for key in {*roles, *secret_names}}
        os.environ.update({k: v for k, v in env.items() if k in roles})
        os.environ.update(secrets or {})
        try:
            dill.loads(raw).run()
            record["stage"] = "COMPLETED"
        except Exception as e:
            record["stage"], record["error"] = "ERROR", str(e)
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
        job = MagicMock()
        job.id, job.url = job_id, f"https://huggingface.co/jobs/{job_id}"
        return job

    def fake_inspect_job(*, job_id, **kwargs):
        job = MagicMock()
        job.status.stage = jobs[job_id]["stage"]
        return job

    return fake_run_uv_job, fake_inspect_job, jobs


def _three_stage_chain(name: str, fail_a: bool = False, **b_kwargs):
    """A -> B -> C, each stage with its own secret."""
    common = {"token": "hf_chaintoken", "poll_interval": 0, "max_retries": 0}
    a = JobsPipelineExecutor(
        pipeline=[_ProbeBlock("A", fail=fail_a)],
        tasks=2,
        logging_dir=f"memory://jobs-chain/{name}/a",
        job_name="A",
        secrets={"SECRET_A": "value-a"},
        **common,
    )
    b = JobsPipelineExecutor(
        pipeline=[_ProbeBlock("B")],
        tasks=2,
        logging_dir=f"memory://jobs-chain/{name}/b",
        job_name="B",
        secrets={"SECRET_B": "value-b"},
        depends=a,
        **common,
        **b_kwargs,
    )
    c = JobsPipelineExecutor(
        pipeline=[_ProbeBlock("C")],
        tasks=1,
        logging_dir=f"memory://jobs-chain/{name}/c",
        job_name="C",
        depends=b,
        **common,
    )
    return a, b, c


def _no_dependency_wait(seconds):
    """Polling sleeps are fine; the minutes-long wait for another launcher's dependency is the bug.

    ``launch_job`` sleeps ``2 * 60`` s per check while it waits on a dependency it did not launch; in a
    single-coordinator chain that wait means the chain is stuck, so the test fails instead of hanging.
    """
    if seconds >= 60:
        raise TimeoutError("would wait for a dependency forever")


def _run_with_hub(executor, hub, live_jobs=(), list_jobs=None):
    fake_run, fake_inspect, _ = hub
    with (
        patch("huggingface_hub.run_uv_job", fake_run),
        patch("huggingface_hub.inspect_job", fake_inspect),
        patch("huggingface_hub.list_jobs", _fake_list_jobs(*live_jobs) if list_jobs is None else list_jobs),
        patch("time.sleep", _no_dependency_wait),
    ):
        executor.run()


class TestJobsExecutor(unittest.TestCase):
    def setUp(self):
        import fsspec

        fs = fsspec.filesystem("memory")
        if fs.exists("/jobs-chain-probe"):
            fs.rm("/jobs-chain-probe", recursive=True)

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
            live = _fake_job({**executor._run_labels, "datatrove_role": "coordinator"}, stage)
            fake_run, submitted = _make_record_only_fake()
            with (
                patch("huggingface_hub.run_uv_job", fake_run),
                patch("huggingface_hub.list_jobs", _fake_list_jobs(live)),
            ):
                executor.run()
            self.assertEqual(len(submitted), expected_submissions, f"{stage} coordinator must not be doubled")

        # a coordinator under another logging_dir is another run, so it must not block this one
        executor = build()
        other = _fake_job({**executor._run_labels, "datatrove_logging_dir": "0" * 12}, "RUNNING")
        fake_run, submitted = _make_record_only_fake()
        with (
            patch("huggingface_hub.run_uv_job", fake_run),
            patch("huggingface_hub.list_jobs", _fake_list_jobs(other)),
        ):
            executor.run()
        self.assertEqual(len(submitted), 1)

    def test_renamed_job_does_not_escape_the_running_check(self):
        """The run is identified by its logging_dir, so renaming it cannot launch a second coordinator."""
        log_dir = get_datafolder("memory://jobs-coordinator/renamed")
        first = JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=1, logging_dir=log_dir, token="t")
        renamed = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=1, logging_dir=log_dir, token="t", job_name="something_else"
        )
        live = _fake_job({**first._run_labels, "datatrove_role": "coordinator"}, "RUNNING")
        fake_run, submitted = _make_record_only_fake()
        with (
            patch("huggingface_hub.run_uv_job", fake_run),
            patch("huggingface_hub.list_jobs", _fake_list_jobs(live)),
        ):
            renamed.run()
        self.assertEqual(len(submitted), 0)

    def test_leftover_workers_block_a_new_launch(self):
        """Workers of an earlier launch read the rank map this launch rewrites, so they must go first."""
        log_dir = get_datafolder("memory://jobs-coordinator/leftover")
        executor = JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=2, logging_dir=log_dir, token="t")
        live = _fake_job({**executor._run_labels, "datatrove_role": "worker", "datatrove_array_index": "0"}, "RUNNING")
        fake_run, submitted = _make_record_only_fake()
        with (
            patch("huggingface_hub.run_uv_job", fake_run),
            patch("huggingface_hub.list_jobs", _fake_list_jobs(live)),
            self.assertRaisesRegex(RuntimeError, "still live") as caught,
        ):
            executor.run()
        self.assertIn("existing-coordinator", str(caught.exception))
        self.assertEqual(len(submitted), 0)
        self.assertFalse(log_dir.isfile("executor.pik"))  # refused before overwriting the previous launch's pickle

    def test_live_worker_of_an_upstream_stage_blocks_the_chain(self):
        """Every stage of the chain is checked, not just the one :meth:`run` was called on."""
        a, b, c = _three_stage_chain("upstream-live")
        live = _fake_job({**a._run_labels, "datatrove_role": "worker"}, "QUEUED")
        hub = _make_chain_hub()
        with self.assertRaisesRegex(RuntimeError, "still live"):
            _run_with_hub(c, hub, live_jobs=[live])
        self.assertEqual(len(hub[2]), 0)

    def test_unlistable_jobs_do_not_launch_anything(self):
        """Without the live-Jobs check a launch could double-run ranks, so a Hub error stops it."""
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=1, logging_dir="memory://jobs-coordinator/unlistable", token="t"
        )

        def unavailable(**kwargs):
            raise ConnectionError("hub unavailable")
            yield  # pragma: no cover -- makes this a generator, so the error surfaces mid-iteration

        fake_run, submitted = _make_record_only_fake()
        with (
            patch("huggingface_hub.run_uv_job", fake_run),
            patch("huggingface_hub.list_jobs", unavailable),
            self.assertRaisesRegex(RuntimeError, "Could not list Jobs") as caught,
        ):
            executor.run()
        self.assertIsInstance(caught.exception.__cause__, ConnectionError)
        self.assertEqual(len(submitted), 0)

    def test_run_returns_the_coordinator_job_id(self):
        """The id is the one thing worth holding on to: it is what `hf jobs logs -f` takes."""
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()], tasks=1, logging_dir="memory://jobs-coordinator/job-id", token="t"
        )
        job_ids = []

        def fake_run_uv_job(script, **kwargs):
            job = MagicMock()
            job.id, job.url = f"job-{len(job_ids)}", "https://huggingface.co/jobs/unused"
            job_ids.append(job.id)
            return job

        with (
            patch("huggingface_hub.run_uv_job", fake_run_uv_job),
            patch("huggingface_hub.list_jobs", lambda **kw: []),
        ):
            self.assertEqual(executor.run(), "job-0")

    def test_the_live_check_filters_by_label_server_side(self):
        """Each stage is looked up by its own run id, so the check is one small labelled request per stage."""
        a, b, c = _three_stage_chain("label-filter")
        hub = _make_chain_hub()
        asked: list[dict] = []
        _run_with_hub(c, hub, list_jobs=_fake_list_jobs(record=asked))

        run_ids = {stage._run_id for stage in (a, b, c)}
        self.assertTrue(all(set(call["labels"]) == {"datatrove_logging_dir"} for call in asked))
        self.assertEqual({call["labels"]["datatrove_logging_dir"] for call in asked}, run_ids)

    def test_coordinator_installs_the_dependencies_of_every_stage(self):
        """Loading the chain's pickle imports every stage, so a light coordinator cannot stay light."""
        a, b, c = _three_stage_chain("dependencies")
        a.dependencies = ["datatrove[io]", "vllm"]
        b.dependencies = ["datatrove[io]", "spacy"]
        hub = _make_chain_hub()
        _run_with_hub(c, hub)
        _, _, jobs = hub

        coordinator = next(job for job in jobs.values() if job["role"] == "coordinator")
        self.assertEqual(coordinator["dependencies"], ["datatrove[io]", "spacy", "vllm"])
        worker_a = next(
            job for job in jobs.values() if job["role"] == "worker" and job["labels"]["datatrove_job_name"] == "A"
        )
        self.assertEqual(worker_a["dependencies"], ["datatrove[io]", "vllm"])  # workers keep their own

    def test_user_labels_cannot_override_the_run_identity(self):
        """The live check reads these labels, so a user label must not be able to forge another run."""
        log_dir = get_datafolder("memory://jobs-coordinator/label-clash")
        executor = JobsPipelineExecutor(
            pipeline=[_EmitBlock()],
            tasks=1,
            logging_dir=log_dir,
            token="t",
            labels={"datatrove_logging_dir": "0" * 12, "team": "nlp"},
        )
        self.assertEqual(executor._run_labels["datatrove_logging_dir"], executor._run_id)
        self.assertEqual(executor._run_labels["team"], "nlp")

        live = _fake_job({**executor._run_labels, "datatrove_role": "coordinator"}, "RUNNING")
        fake_run, submitted = _make_record_only_fake()
        with (
            patch("huggingface_hub.run_uv_job", fake_run),
            patch("huggingface_hub.list_jobs", _fake_list_jobs(live)),
        ):
            executor.run()
        self.assertEqual(len(submitted), 0)

    def test_one_coordinator_runs_the_whole_chain(self):
        """`run()` on the last stage submits one coordinator, which runs each stage after the previous one."""
        a, b, c = _three_stage_chain("in-order")
        hub = _make_chain_hub()
        _run_with_hub(c, hub)
        _, _, jobs = hub

        coordinators = [job for job in jobs.values() if job["role"] == "coordinator"]
        self.assertEqual([job["labels"]["datatrove_job_name"] for job in coordinators], ["C"])
        self.assertEqual(coordinators[0]["stage"], "COMPLETED")
        probes = _probes()
        self.assertEqual(sorted(probes), ["A-0", "A-1", "B-0", "B-1", "C-0"])
        # every rank of a stage ran before any rank of the next one
        order = sorted(probes, key=lambda probe: probes[probe]["order"])
        self.assertEqual([probe[0] for probe in order], ["A", "A", "B", "B", "C"])
        for stage in (a, b, c):
            self.assertTrue(stage.logging_dir.isfile("stats.json"))

    def test_chain_secrets_reach_each_stage_and_no_pickle(self):
        """Each stage's workers get that stage's secrets; no pickle any Job reads holds a credential."""
        a, b, c = _three_stage_chain("secrets")
        hub = _make_chain_hub()
        _run_with_hub(c, hub)
        _, _, jobs = hub

        probes = _probes()
        self.assertEqual({probes[f"A-{r}"]["secret"] for r in (0, 1)}, {"value-a"})
        self.assertEqual({probes[f"B-{r}"]["secret"] for r in (0, 1)}, {"value-b"})
        for job in jobs.values():
            for credential in (b"hf_chaintoken", b"value-a", b"value-b"):
                self.assertNotIn(credential, job["pik"])

    def test_chain_rejects_one_secret_name_with_two_values(self):
        """The coordinator receives every stage's secrets as env vars, so a name cannot carry two values."""
        a, b, c = _three_stage_chain("secret-clash")
        b.secrets = {"SECRET_A": "another-value"}
        with self.assertRaises(ValueError):
            _run_with_hub(c, _make_chain_hub())

    def test_chain_rejects_two_tokens(self):
        """Each stage would otherwise run with whichever token the coordinator got."""
        a, b, c = _three_stage_chain("token-clash")
        a.token = "hf_othertoken"
        with self.assertRaises(ValueError):
            _run_with_hub(c, _make_chain_hub())

    def test_failed_rerun_is_not_hidden_by_old_markers(self):
        """A dependency rerun that fails stops the chain, even with completion markers from an earlier run."""
        a, b, c = _three_stage_chain("stale-markers")
        _run_with_hub(c, _make_chain_hub())
        a.skip_completed = False  # recompute A: its old completions/ markers stay in place
        a.pipeline[0].fail = True
        hub = _make_chain_hub()
        _run_with_hub(c, hub)
        _, _, jobs = hub

        coordinator = next(job for job in jobs.values() if job["role"] == "coordinator")
        self.assertEqual(coordinator["stage"], "ERROR")

    def test_failed_dependency_fails_the_coordinator(self):
        """A stage with failed tasks stops the chain and the coordinator Job ends as ERROR, without waiting."""
        a, b, c = _three_stage_chain("fails", fail_a=True)
        hub = _make_chain_hub()
        _run_with_hub(c, hub)
        _, _, jobs = hub

        coordinator = next(job for job in jobs.values() if job["role"] == "coordinator")
        self.assertEqual(coordinator["stage"], "ERROR")
        self.assertIn("'A'", coordinator["error"])
        self.assertEqual(_probes(), {})  # neither B nor C ran

    def test_run_on_dependency_fail_continues_the_chain(self):
        a, b, c = _three_stage_chain("continues", fail_a=True, run_on_dependency_fail=True)
        _run_with_hub(c, _make_chain_hub())
        self.assertIn("B-0", _probes())

    def test_rerun_after_a_failure_restarts_the_failed_stage(self):
        """Rerunning the same objects resumes from the failed stage instead of waiting on it."""
        a, b, c = _three_stage_chain("rerun", fail_a=True)
        _run_with_hub(c, _make_chain_hub())
        a.pipeline[0].fail = False  # the user fixes stage A and reruns the same script objects
        _run_with_hub(c, _make_chain_hub())
        self.assertEqual(sorted(_probes()), ["A-0", "A-1", "B-0", "B-1", "C-0"])

    def test_env_coordinator_key_rejected(self):
        """The coordinator role variable is reserved, like the array-index one."""
        log_dir = get_datafolder("memory://jobs-coordinator/env-reserved")
        with self.assertRaises(ValueError):
            JobsPipelineExecutor(pipeline=[_EmitBlock()], tasks=1, logging_dir=log_dir, env={COORDINATOR_ENV_VAR: "1"})


if __name__ == "__main__":
    unittest.main()
