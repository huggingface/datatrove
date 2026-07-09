from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from collections import deque
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import dill
from dill import CONTENTS_FMODE

from datatrove.executor.base import DistributedEnvVars, PipelineExecutor
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import add_task_logger, close_task_logger, log_pipeline, logger
from datatrove.utils.stats import PipelineStats


# Set on each launched Job to tell the worker which slice of `ranks_to_run.json` it owns.
ARRAY_INDEX_ENV_VAR = "DATATROVE_JOB_ARRAY_INDEX"
# Mirror of huggingface_hub._jobs_api.TERMINAL_JOB_STAGES, duplicated to avoid importing a private symbol.
_TERMINAL_JOB_STAGES = ("COMPLETED", "ERROR", "CANCELED", "DELETED")
_SUCCESS_JOB_STAGE = "COMPLETED"


class JobsPipelineExecutor(PipelineExecutor):
    """Execute a pipeline on Hugging Face Jobs.

    **Experimental**: this executor is experimental — it may change, break, or be removed
    in future releases without the usual deprecation cycle.

    Fans a datatrove pipeline out across a pool of Hugging Face Jobs, mirroring
    :class:`~datatrove.executor.slurm.SlurmPipelineExecutor` but launching Jobs with
    ``huggingface_hub.run_uv_job`` instead of calling ``sbatch``.

    Like the Slurm executor, :meth:`run` is two-phase: when it detects it is running
    *inside* a launched Job (the ``DATATROVE_JOB_ARRAY_INDEX`` env var is set) it runs its
    assigned block of ranks; otherwise it launches the Jobs. All coordination happens
    through the shared fsspec ``logging_dir`` (``executor.pik``, ``ranks_to_run.json``,
    ``completions/``, ``stats/``), so it MUST point to a *remote* folder that both the
    launching machine and the Jobs can read/write (e.g. ``hf://buckets/<user>/<bucket>/...``
    (huggingface_hub>=1.6.0), ``hf://datasets/<repo>`` or ``s3://<bucket>``).

    Each Job runs ``launch_pickled_pipeline <logging_dir>/executor.pik`` in a ``uv``
    environment built from ``dependencies``; that rehydrates this executor and calls
    :meth:`run`, which — with ``DATATROVE_JOB_ARRAY_INDEX`` set — runs the Job's block of
    ranks. Concurrency (Slurm's ``%workers`` array throttle) is enforced by a local polling
    loop that keeps at most ``workers`` Jobs alive at a time.

    Because completion is tracked with ``completions/{rank:05d}`` marker files, reruns are
    idempotent: already-completed ranks are skipped and only the Jobs for the remaining
    ranks are relaunched. This replaces Slurm's requeue / ``SIGUSR1`` handling, which has no
    Jobs equivalent. Before rerunning, make sure no Jobs from a previous launch are still
    queued or running (cancel them first): each launch rewrites ``executor.pik`` and
    ``ranks_to_run.json``, which stale Jobs would then read.

    Note: the launching process (the "coordinator") currently runs **locally** and stays alive for
    the whole run — it submits the Jobs, polls them to honor ``workers``, and merges stats at the
    end (the same launch-and-block model as Local/Ray; only Slurm is fire-and-forget). Only the
    compute is remote. If the coordinator is interrupted, in-flight Jobs still finish and record
    their ranks; just rerun to launch the rest (resume is idempotent). Running the coordinator
    itself inside a Job, so nothing needs to stay on the launching machine, is a possible future
    enhancement.

    [!] do not launch Jobs from within a Job.

    Args:
        pipeline: a list of PipelineStep and/or custom functions with arguments
            (data: DocumentsPipeline, rank: int, world_size: int).
        tasks: total number of tasks (the world_size to shard the pipeline over).
        workers: how many Jobs to run simultaneously. -1 (default) launches them all at
            once. This is the equivalent of Slurm's ``%workers`` array throttle, enforced
            here by a local polling loop.
        logging_dir: where to save logs, stats, completions and the pickled executor.
            Must resolve to a *remote* datatrove.io.DataFolder (``hf://`` or ``s3://``).
        flavor: Hugging Face Jobs hardware flavor for each Job (default ``"cpu-basic"``).
        image: optional Docker image for the Jobs. Defaults to the huggingface_hub uv
            image. A custom image must have ``uv`` installed.
        dependencies: pip requirements installed in each Job's ``uv`` environment (passed
            as ``uv run --with``). Must include datatrove and anything your pipeline steps
            import (e.g. ``"datasets"``). Defaults to ``["datatrove[io]"]``; until this
            executor is released, point datatrove at the branch, e.g.
            ``["datatrove[io] @ git+https://github.com/<user>/datatrove@<branch>", "datasets"]``.
        timeout: per-Job timeout, as seconds (int) or a string like ``"2h"`` / ``"30m"``.
        tasks_per_job: how many datatrove tasks each Job runs. Reduces the number of Jobs
            launched (default 1).
        depends: another JobsPipelineExecutor that must finish before this one starts.
        skip_completed: whether to skip tasks completed in previous runs (default True).
        randomize_start_duration: max seconds to randomly delay the start of each task.
        python: Python version for the Job's ``uv`` environment (e.g. ``"3.12"``).
        namespace: Hugging Face namespace to launch the Jobs under (defaults to the
            authenticated user).
        labels: extra labels attached to each Job.
        token: Hugging Face token. Defaults to the locally saved login. Passed to each Job
            as the ``HF_TOKEN`` secret so it can read/write the logging_dir.
        secrets: extra secrets passed to each Job (merged with the ``HF_TOKEN`` secret; an
            ``HF_TOKEN`` key here overrides the resolved token).
        max_retries: how many times to relaunch a Job that ends in a non-success state
            within a single run (default 1). Ranks still failed after that are logged and
            left incomplete — :meth:`run` does NOT raise; rerun to resume. A dependent
            executor (``depends=``) fails fast on them instead.
        poll_interval: seconds between polls of the running Jobs (default 15).
        run_on_dependency_fail: if a ``depends`` job finishes with failed (permanently incomplete) tasks,
            continue anyway instead of raising. Default False (fail fast, like Slurm's ``afterok``).
        job_name: name used for logging and as a Job label (default ``"data_processing"``).

    Example:
        ```python
        from datatrove.executor import JobsPipelineExecutor
        from datatrove.pipeline.readers import HuggingFaceDatasetReader
        from datatrove.pipeline.writers import ParquetWriter

        JobsPipelineExecutor(
            pipeline=[
                HuggingFaceDatasetReader("stanfordnlp/imdb", dataset_options={"split": "train"}),
                ParquetWriter("hf://datasets/<user>/imdb-processed/data"),
            ],
            tasks=10,
            workers=4,
            logging_dir="hf://datasets/<user>/imdb-processed/logs",
            dependencies=["datatrove[io]", "datasets"],
            flavor="cpu-basic",
        ).run()
        ```
    """

    def __init__(
        self,
        pipeline: list[PipelineStep | Callable],
        tasks: int,
        workers: int = -1,
        logging_dir: DataFolderLike = None,
        flavor: str = "cpu-basic",
        image: str | None = None,
        dependencies: list[str] | None = None,
        timeout: int | float | str | None = None,
        tasks_per_job: int = 1,
        depends: "JobsPipelineExecutor | None" = None,
        skip_completed: bool = True,
        randomize_start_duration: int = 0,
        python: str | None = None,
        namespace: str | None = None,
        labels: dict[str, str] | None = None,
        token: str | None = None,
        secrets: dict[str, Any] | None = None,
        max_retries: int = 1,
        poll_interval: int = 15,
        run_on_dependency_fail: bool = False,
        job_name: str = "data_processing",
    ):
        logger.warning("JobsPipelineExecutor is experimental: it may change, break, or be removed in future releases.")
        super().__init__(pipeline, logging_dir, skip_completed, randomize_start_duration)
        if self.logging_dir.is_local():
            raise ValueError(
                "JobsPipelineExecutor requires a remote logging_dir (e.g. hf://datasets/<repo> or "
                "s3://<bucket>) that both the launching machine and the Jobs can access. Got a local "
                f"path: {self.logging_dir.path!r}."
            )
        if tasks_per_job < 1:
            raise ValueError(f"tasks_per_job must be >= 1, got {tasks_per_job}.")
        if workers != -1 and workers < 1:
            raise ValueError(f"workers must be -1 (unlimited) or >= 1, got {workers}.")
        self.tasks = tasks
        self.workers = workers
        self.flavor = flavor
        self.image = image
        self.dependencies = dependencies if dependencies is not None else ["datatrove[io]"]
        self.timeout = timeout
        self.tasks_per_job = tasks_per_job
        self.depends = depends
        self.python = python
        self.namespace = namespace
        self.labels = labels
        self.token = token
        self.secrets = secrets
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.run_on_dependency_fail = run_on_dependency_fail
        self.job_name = job_name
        self._launched = False

    def run(self):
        """Run the pipeline.

        Two-phase: inside a launched Job (``DATATROVE_JOB_ARRAY_INDEX`` set) run the block of
        ranks assigned to that Job; otherwise launch the Jobs.
        """
        if ARRAY_INDEX_ENV_VAR in os.environ:
            self._run_array_index(int(os.environ[ARRAY_INDEX_ENV_VAR]))
        else:
            self.launch_job()

    def _run_array_index(self, array_index: int):
        """Run the block of ranks assigned to one Job, identified by its array index."""
        with self.logging_dir.open("ranks_to_run.json", "r") as f:
            all_ranks = json.load(f)
        start = array_index * self.tasks_per_job
        if start >= len(all_ranks):
            logger.info(f"Array index {array_index} has no ranks to run.")
            return
        # deepcopy the pipeline before each rank so per-rank stats and step state don't accumulate
        # across ranks that share this Job's process (mirrors LocalPipelineExecutor's workers==1 path).
        base_pipeline = self.pipeline
        for i in range(start, min(start + self.tasks_per_job, len(all_ranks))):
            self.pipeline = deepcopy(base_pipeline)
            self._run_for_rank(all_ranks[i])

    def launch_job(self):
        """Pickle this executor and fan it out across a pool of Hugging Face Jobs."""
        from huggingface_hub import get_token

        assert not self.depends or isinstance(self.depends, JobsPipelineExecutor), (
            "depends= must be a JobsPipelineExecutor"
        )
        if self.depends:
            # launch any unlaunched dependency, then wait for its completions
            depends_launched_here = not self.depends._launched
            if depends_launched_here:
                logger.info(f'Launching dependency job "{self.depends.job_name}"')
                self.depends.launch_job()
            while True:
                # completions/ is written by the remote Jobs, so drop the launcher's cached directory
                # listing before checking — a long-lived HfFileSystem otherwise keeps serving a stale
                # (pre-completion) listing and we would never see the Jobs finish.
                self.depends.logging_dir.fs.invalidate_cache()
                incomplete = self.depends.get_incomplete_ranks(skip_completed=True)
                if not incomplete:
                    break
                if depends_launched_here:
                    # launch_job() blocks, so the dependency has finished; any ranks still incomplete have
                    # failed (retries exhausted). Don't wait forever.
                    if self.run_on_dependency_fail:
                        logger.warning(
                            f"Dependency '{self.depends.job_name}' finished with {len(incomplete)} failed "
                            f"task(s); continuing anyway (run_on_dependency_fail=True)."
                        )
                        break
                    raise RuntimeError(
                        f"Dependency job '{self.depends.job_name}' finished with {len(incomplete)}/"
                        f"{self.depends.world_size} tasks still incomplete. Rerun to resume, raise max_retries, "
                        f"or set run_on_dependency_fail=True to continue anyway."
                    )
                # dependency was launched by another process and is still running — keep waiting
                logger.info(f"Dependency job still has {len(incomplete)}/{self.depends.world_size} tasks. Waiting...")
                time.sleep(2 * 60)
            self.depends = None  # avoid pickling the dependency chain

        ranks_to_run = self.get_incomplete_ranks()
        if len(ranks_to_run) == 0:
            logger.info(f"Skipping launch of {self.job_name} as all {self.tasks} tasks have already been completed.")
            self._launched = True
            if not self.logging_dir.isfile("stats.json"):
                self._merge_stats()  # a previous coordinator may have died after the last rank completed
            return

        # pickle ourselves; each Job rehydrates this via `launch_pickled_pipeline` and calls run()
        executor = deepcopy(self)
        # never persist credentials into the (shared) logging_dir: the Jobs get HF_TOKEN and any user
        # secrets as Job secrets instead (see _submit_array_index)
        executor.token = None
        executor.secrets = None
        with self.logging_dir.open("executor.pik", "wb") as executor_f:
            dill.dump(executor, executor_f, fmode=CONTENTS_FMODE)
        self.save_executor_as_json()
        with self.logging_dir.open("ranks_to_run.json", "w") as ranks_to_run_file:
            # saved once to avoid races: array index i owns ranks_to_run[i*tasks_per_job:(i+1)*tasks_per_job]
            json.dump(ranks_to_run, ranks_to_run_file)

        nb_jobs = math.ceil(len(ranks_to_run) / self.tasks_per_job)
        max_concurrent = self.workers if self.workers != -1 else nb_jobs
        pik_path = self.logging_dir.resolve_paths("executor.pik")
        token = self.token or get_token()
        if token is None:
            raise ValueError(
                "No Hugging Face token found. Log in with `hf auth login` or pass token=... so the Jobs "
                "can read/write the logging_dir."
            )

        logger.info(
            f"Launching {nb_jobs} Hugging Face Job(s) for '{self.job_name}' ({len(ranks_to_run)} tasks, "
            f"{self.tasks_per_job} per job, up to {max_concurrent} concurrent) on flavor '{self.flavor}'."
        )
        self._launched = True
        self._launch_and_wait(nb_jobs, pik_path, max_concurrent, token)
        self._merge_stats()

    def save_executor_as_json(self, indent: int = 4):
        """Same as the base version, but never persists credentials into the (shared) logging_dir."""
        token, secrets = self.token, self.secrets
        self.token = self.secrets = None
        try:
            super().save_executor_as_json(indent=indent)
        finally:
            self.token, self.secrets = token, secrets

    def _submit_array_index(self, array_index: int, pik_path: str, token: str) -> str:
        """Launch a single Hugging Face Job for one array index; returns its job id."""
        from huggingface_hub import run_uv_job

        from datatrove.tools import launch_pickled_pipeline

        job = run_uv_job(
            launch_pickled_pipeline.__file__,  # uploaded + run as `uv run launch_pickled_pipeline.py <pik>`
            script_args=[pik_path],
            dependencies=self.dependencies,
            python=self.python,
            image=self.image,
            env={
                ARRAY_INDEX_ENV_VAR: str(array_index),
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "PYTHONUNBUFFERED": "1",
            },
            secrets={"HF_TOKEN": token, **(self.secrets or {})},
            flavor=self.flavor,
            timeout=self.timeout,
            labels={
                "datatrove_job_name": self.job_name,
                "datatrove_array_index": str(array_index),
                **(self.labels or {}),
            },
            namespace=self.namespace,
            token=token,
        )
        logger.info(f"Launched array index {array_index} as Job {job.id} ({job.url}).")
        return job.id

    def _launch_and_wait(self, nb_jobs: int, pik_path: str, max_concurrent: int, token: str):
        """Local window-manager: keep at most ``max_concurrent`` Jobs live until every index finishes."""
        from huggingface_hub import inspect_job

        pending: deque[int] = deque(range(nb_jobs))
        running: dict[str, int] = {}  # job_id -> array_index
        retries: dict[int, int] = {}
        failed: list[int] = []

        def fill():
            while pending and len(running) < max_concurrent:
                idx = pending.popleft()
                running[self._submit_array_index(idx, pik_path, token)] = idx

        fill()
        while running:
            time.sleep(self.poll_interval)
            for job_id in list(running):
                stage = inspect_job(job_id=job_id, namespace=self.namespace, token=token).status.stage
                if stage not in _TERMINAL_JOB_STAGES:
                    continue
                idx = running.pop(job_id)
                if stage == _SUCCESS_JOB_STAGE:
                    logger.info(f"Job {job_id} (array index {idx}) completed.")
                elif retries.get(idx, 0) < self.max_retries:
                    retries[idx] = retries.get(idx, 0) + 1
                    logger.warning(
                        f"Job {job_id} (array index {idx}) ended as {stage}; "
                        f"resubmitting ({retries[idx]}/{self.max_retries})."
                    )
                    pending.append(idx)
                else:
                    logger.error(
                        f"Job {job_id} (array index {idx}) failed as {stage} after {retries.get(idx, 0)} retries."
                    )
                    failed.append(idx)
            fill()

        if failed:
            logger.warning(
                f"{len(failed)} Job(s) failed after retries (array indices {failed}). Incomplete ranks will be "
                f"retried on the next run (completed ranks are skipped)."
            )

    def _merge_stats(self):
        """Merge per-rank stats into ``stats.json`` once every task is complete."""
        # the Jobs wrote completions/ + stats/ remotely; refresh the launcher's cached listing first
        self.logging_dir.fs.invalidate_cache()
        # check the actual completion markers even when skip_completed=False
        incomplete = self.get_incomplete_ranks(range(self.world_size), skip_completed=True)
        if incomplete:
            logger.warning(
                f"{len(incomplete)}/{self.world_size} tasks still incomplete; not merging stats. Rerun to resume."
            )
            return
        total_stats = PipelineStats()
        for rank in range(self.world_size):
            with self.logging_dir.open(f"stats/{rank:05d}.json", "r") as f:
                total_stats += PipelineStats.from_json(json.load(f))
        with self.logging_dir.open("stats.json", "wt") as f:
            total_stats.save_to_disk(f)
        logger.success(total_stats.get_repr(f"All {self.world_size} tasks."))

    def _run_for_rank(self, rank: int, local_rank: int = 0, node_rank: int = -1) -> PipelineStats:
        """Run one rank, logging to a local tempdir and uploading the log to ``logging_dir`` on finish.

        Overrides the base method because the remote (object-store) ``logging_dir`` is a poor target for
        the incrementally-written task log; per-rank stats and the completion marker are still written
        straight to ``logging_dir``. Mirrors ``RayPipelineExecutor._run_for_rank``.
        """
        if self.is_rank_completed(rank):
            logger.info(f"Skipping {rank=} as it has already been completed.")
            return PipelineStats()

        self._set_distributed_environment(node_rank)
        if self.randomize_start_duration > 0:
            time.sleep(random.randint(0, self.randomize_start_duration))

        # log locally and upload the log to logging_dir once the pipeline finishes
        local_logs_dir = get_datafolder(f"{tempfile.gettempdir()}/datatrove_jobs_logs")
        logfile = add_task_logger(local_logs_dir, rank, local_rank, node_rank=node_rank)
        log_pipeline(self.pipeline)

        stats = PipelineStats()
        try:
            # pipe data from one step to the next
            pipelined_data = None
            for pipeline_step in self.pipeline:
                if callable(pipeline_step):
                    pipelined_data = pipeline_step(pipelined_data, rank, self.world_size)
                elif isinstance(pipeline_step, Sequence) and not isinstance(pipeline_step, str):
                    pipelined_data = pipeline_step
                else:
                    raise ValueError
            if pipelined_data:
                deque(pipelined_data, maxlen=0)

            logger.success(f"Processing done for {rank=}")

            stats = PipelineStats(self.pipeline)
            if node_rank <= 0:
                with self.logging_dir.open(f"stats/{rank:05d}.json", "w") as f:
                    stats.save_to_disk(f)
                logger.info(stats.get_repr(f"Task {rank}"))
                self.mark_rank_as_completed(rank)
        except Exception as e:
            logger.exception(e)
            raise e
        finally:
            close_task_logger(logfile)
            with (
                local_logs_dir.open(f"logs/task_{rank:05d}.log", "r") as f,
                self.logging_dir.open(f"logs/task_{rank:05d}.log", "w") as f_out,
            ):
                f_out.write(f.read())
        return stats

    def get_distributed_env(self, node_rank: int = -1) -> DistributedEnvVars:
        """Distributed env vars for the JOBS executor (each Job is a single node)."""
        return DistributedEnvVars(
            datatrove_node_ips="localhost",
            datatrove_cpus_per_task="-1",
            datatrove_mem_per_cpu="-1",
            datatrove_gpus_on_node="-1",
            datatrove_executor="JOBS",
        )

    @property
    def world_size(self) -> int:
        return self.tasks
