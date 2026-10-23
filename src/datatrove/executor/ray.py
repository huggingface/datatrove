#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import time
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from datatrove.executor.base import DistributedEnvVars, PipelineExecutor
from datatrove.io import DataFolderLike, file_is_local, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.logging import add_task_logger, close_task_logger, log_pipeline, logger
from datatrove.utils.stats import PipelineStats


# how long a multi-node placement group may take to become ready before the job fails (unsatisfiable requests must not hang the driver)
PG_READY_TIMEOUT_S = 300
# how many times a job may be resubmitted after a retriable failure (1 initial attempt + MAX_RESUBMITS)
MAX_RESUBMITS = 3


def convert_remote_options_to_bundle(remote_options: dict) -> dict:
    """Converts remote options (num_cpus/memory/num_gpus) to a bundle for a placement group.
    Args: remote_options: executor resource options. Returns: a bundle (CPU/memory/GPU keys) with the set options."""
    return {
        key: remote_options[source]
        for source, key in (("num_cpus", "CPU"), ("memory", "memory"), ("num_gpus", "GPU"))
        if source in remote_options
    }


def run_rank_task(executor: RayPipelineExecutor, rank: int, local_rank: int) -> None:
    """Single-node entry point: one native Ray task per rank — no actors, no pools, no threads.
    Args: executor: executor to run; rank: global rank; local_rank: rank within the job (only used for logging)."""
    time.sleep(random.randint(0, executor.randomize_start_duration))
    executor._run_for_rank(rank, local_rank if executor.log_first else local_rank + 1, node_rank=-1)


class RankWorker:
    """Multi-node gang member (only used when nodes_per_task > 1): runs the job's ranks sequentially on its node."""

    def get_node_ip(self) -> str:
        """Return this actor's node IP address. Returns: the node IP, used to fill RAY_NODELIST for the gang."""
        import ray

        return ray.util.get_node_ip_address()

    def run_for_rank(
        self, executor: RayPipelineExecutor, ranks: list[int], node_ips: list[str], node_idx: int
    ) -> None:
        """Run all of the job's ranks sequentially on this node (node 0's runs are the ones writing stats/markers).
        Args: executor: executor to run; ranks: the job's global ranks (every gang member gets the full list);
            node_ips: IPs of all gang nodes, in bundle order; node_idx: this actor's index (0 = master node)."""
        import os

        os.environ["RAY_NODELIST"], os.environ["RAY_NODEID"] = ",".join(node_ips), str(node_idx)
        time.sleep(random.randint(0, executor.randomize_start_duration))
        for local_rank, rank in enumerate(ranks):
            # node 0 is the master and the only one writing stats/completion markers (node_rank <= 0 in _run_for_rank)
            executor._run_for_rank(rank, local_rank if executor.log_first else local_rank + 1, node_rank=node_idx)


@dataclass
class RankJob:
    """One unit of work (a `tasks_per_job` batch of ranks) with its current attempt's in-flight refs and bookkeeping."""

    ranks: list[int]
    attempts: int = 1
    deadline: float | None = None  # monotonic deadline for the current attempt; None = no time limit
    refs: tuple = ()  # in-flight Ray tasks of the current attempt
    workers: tuple = ()  # gang actors (multi-node only)
    pg: object | None = None  # placement group (multi-node only)
    error: BaseException | None = None  # first failure observed in the current attempt
    retriable: bool = True  # whether `error` may be retried within the resubmission budget
    reason: str = ""  # filled in when the job permanently fails (used by the final failure summary)


def _retriable_ray_errors() -> tuple[type[Exception], ...]:
    """Ray-side infrastructure failures worth another attempt (lazily imported: ray is an optional dependency).
    Returns: tuple of exception types that `run` treats as retriable."""
    from ray.exceptions import ObjectLostError, RayActorError, TaskCancelledError, WorkerCrashedError

    return (WorkerCrashedError, ObjectLostError, RayActorError, TaskCancelledError)


class RayPipelineExecutor(PipelineExecutor):
    """
    Executor to run a pipeline using Ray. Expects the Ray cluster to be set up already (e.g., via `ray.init()`).
    Single-node (nodes_per_task=1) runs one native Ray task per rank; multi-node runs a gang of node actors.
    Args:
        pipeline: a list of PipelineStep and/or custom callables with arguments (data, rank, world_size)
        tasks: total number of tasks to run the pipeline on (default: 1); workers: maximum number of jobs in flight
            (each job batches tasks_per_job ranks) (default: -1 = no limit)
        depends: another RayPipelineExecutor that should run before this one; skip_completed: whether to skip tasks
            that were completed in previous runs (default: True)
        logging_dir: where to save logs, stats, etc. Should be parsable into a datatrove.io.DataFolder
        randomize_start_duration: the maximum number of seconds to delay the start of each task
        cpus_per_task: CPUs per task (default: 1); mem_per_cpu_gb: memory (GB) per CPU (default: 2); gpus_per_task: GPUs per task (default: 0)
        nodes_per_task: if > 1, a STRICT_SPREAD placement group holds one actor per node, each running the job's ranks
            (node 0 writes stats/markers). Default: 1
        ray_remote_kwargs: additional kwargs to pass to the ray.remote options; log_first: whether the first task in a
            ray job should log to console (default: False)
        tasks_per_job: number of ranks per Ray job (default: 1); time: time limit in seconds per job attempt (None = no limit)
    """

    def __init__(
        self,
        pipeline: list[PipelineStep | Callable],
        tasks: int = 1,
        workers: int = -1,
        depends: "RayPipelineExecutor" = None,
        skip_completed: bool = True,
        logging_dir: DataFolderLike = None,
        randomize_start_duration: int = 0,
        cpus_per_task: int = 1,
        mem_per_cpu_gb: float = 2,
        gpus_per_task: int = 0,
        nodes_per_task: int = 1,
        ray_remote_kwargs: dict = None,
        log_first: bool = False,
        tasks_per_job: int = 1,
        time: int | None = None,
    ):
        # Check if the logging_dir is local fs and if so issue a warning that for synchronization it has to be a shared filesystem
        if logging_dir and file_is_local(logging_dir):
            logger.warning(
                "Logging directory points to a local filesystem. For correct synchronization to work this "
                "filesystem needs be shared across the submitting node as well as the workers and needs "
                "to be persistent across node restarts."
            )

        super().__init__(pipeline, logging_dir, skip_completed, randomize_start_duration)
        self.tasks = tasks
        self.workers = workers if workers != -1 else tasks
        self.depends = depends
        self.cpus_per_task = cpus_per_task
        self.gpus_per_task = gpus_per_task
        self.mem_per_cpu_gb = mem_per_cpu_gb
        self.ray_remote_kwargs = ray_remote_kwargs
        self.tasks_per_job = tasks_per_job
        self.log_first, self.time, self._launched = log_first, time, False
        self.nodes_per_task = nodes_per_task

    def get_distributed_env(self, node_rank: int = -1) -> DistributedEnvVars:
        """Get distributed environment variables for RAY executor.
        Args: node_rank: unused here (node IPs come from the RAY_NODELIST env var; kept for the base-class signature).
        Returns: DistributedEnvVars dictionary with all required environment variables."""
        import os

        node_ips = os.environ.get("RAY_NODELIST", "")

        return DistributedEnvVars(
            datatrove_node_ips=node_ips,
            datatrove_cpus_per_task=str(self.cpus_per_task),
            datatrove_mem_per_cpu=str(self.mem_per_cpu_gb),
            datatrove_gpus_on_node=str(self.gpus_per_task),
            datatrove_executor="RAY",
        )

    @property
    def world_size(self) -> int:
        """Total number of ranks to run the pipeline on (always equal to `tasks`)."""
        return self.tasks

    def _create_placement_group(self, remote_options: dict, ranks: list[int]) -> object:
        """Create a STRICT_SPREAD placement group (one bundle per node) and wait for readiness with a deadline.
        Args: remote_options: executor resource options (converted to one bundle per node); ranks: the job's global
            ranks (used in the error message).
        Returns: the ready placement group. Raises: RuntimeError if not ready within PG_READY_TIMEOUT_S seconds."""
        import ray

        bundle = convert_remote_options_to_bundle(remote_options)
        pg = ray.util.placement_group([bundle] * self.nodes_per_task, strategy="STRICT_SPREAD")
        deadline = time.monotonic() + PG_READY_TIMEOUT_S
        while not ray.wait([pg.ready()], timeout=max(0.1, min(10, deadline - time.monotonic())))[0]:
            if time.monotonic() > deadline:
                ray.util.remove_placement_group(pg)
                raise RuntimeError(
                    f"Placement group for ranks {ranks} ({self.nodes_per_task} nodes, strategy STRICT_SPREAD) was not ready within "
                    f"{PG_READY_TIMEOUT_S}s. Check that the cluster has enough nodes and free resources.",
                )
        return pg

    def _launch_gang(self, executor_ref: RayPipelineExecutor, job: RankJob, remote_options: dict) -> None:
        """One actor per node behind a STRICT_SPREAD placement group; each processes the job's ranks, node 0 writes.
        Args: executor_ref: ObjectRef to the pickled executor; job: the job to launch (pg/workers/refs fields are
            filled in); remote_options: executor resource options for the group and actors."""
        import ray

        try:
            job.pg = self._create_placement_group(remote_options, job.ranks)
            job.workers = tuple(
                ray.remote(RankWorker)
                .options(placement_group=job.pg, placement_group_bundle_index=i, **remote_options)
                .remote()
                for i in range(self.nodes_per_task)
            )
            node_ips = ray.get([worker.get_node_ip.remote() for worker in job.workers])
            job.refs = tuple(
                w.run_for_rank.remote(executor_ref, job.ranks, node_ips, i) for i, w in enumerate(job.workers)
            )
        except BaseException:  # also release gang resources on KeyboardInterrupt during the blocking ray.get
            self._release_job(job, job.refs)
            raise

    def _release_job(self, job: RankJob, refs: tuple = ()) -> None:
        """Force-cancel `refs` (bounded), kill gang actors and remove the placement group. Best effort, idempotent.
        Args: job: the job whose gang resources should be released; refs: in-flight refs to cancel (default: none)."""
        import ray

        if refs:
            ray.cancel(*refs, force=True, recursive=True)
            ray.wait(list(refs), num_returns=len(refs), timeout=5)  # bounded: cancellation is asynchronous
        try:
            for worker in job.workers:
                ray.kill(worker)
            if job.pg is not None:
                ray.util.remove_placement_group(job.pg)
        except Exception as e:
            logger.warning(f"Failed to release gang resources for ranks {job.ranks}: {e}")
        job.refs, job.workers, job.pg = (), (), None

    def run(self):
        """Run the pipeline for each rank using Ray tasks (single-node: one native Ray task per rank; multi-node: gang).
        A job that ultimately fails (application error, exhausted retries or timeout) makes run() raise, not just warn."""
        check_required_dependencies("ray", ["ray"])
        import ray

        assert not self.depends or (isinstance(self.depends, RayPipelineExecutor)), (
            "depends= must be a RayPipelineExecutor"
        )
        if self.depends:
            # take care of launching any unlaunched dependencies
            if not self.depends._launched:
                logger.info(f'Launching dependency job "{self.depends}"')
                self.depends.run()
            # set skip_completed=True to get *real* incomplete task count
            while (incomplete := len(self.depends.get_incomplete_ranks(skip_completed=True))) > 0:
                logger.info(f"Dependency job still has {incomplete}/{self.depends.world_size} tasks. Waiting...")
                time.sleep(2 * 60)

        self._launched = True
        incomplete_ranks = self.get_incomplete_ranks(range(self.world_size))
        if not incomplete_ranks:
            logger.info(f"All {self.world_size} tasks appear to be completed already. Nothing to run.")
            return

        logger.info(f"Will run pipeline on {len(incomplete_ranks)} incomplete ranks out of {self.world_size} total.")

        self.save_executor_as_json()

        executor_ref = ray.put(self)
        remote_options = {
            "num_cpus": self.cpus_per_task,
            "num_gpus": self.gpus_per_task,
            "memory": int(self.mem_per_cpu_gb * self.cpus_per_task * 1024 * 1024 * 1024),
        }
        remote_options.update(self.ray_remote_kwargs or {})
        # max_retries=0: the loop below owns the retry budget, ray must not silently re-run failed ranks
        remote_run = ray.remote(run_rank_task).options(max_retries=0, **remote_options)

        size = self.tasks_per_job
        queue = deque(RankJob(ranks=incomplete_ranks[i : i + size]) for i in range(0, len(incomplete_ranks), size))
        total, completed = len(queue), 0
        pending: dict = {}  # ref -> job, single owner of refs that have not settled yet
        active, failed = [], []  # in-flight jobs; jobs settled as permanently failed

        def submit(job: RankJob) -> None:
            """(Re)start a job: one native Ray task per rank, or a gang of node actors behind a STRICT_SPREAD group."""
            job.error, job.retriable, job.reason = None, True, ""
            job.deadline = time.monotonic() + self.time if self.time else None
            # available_resources() no longer counts what our own in-flight tasks hold; credit it back (all tasks
            # reserve the same shape). limitation: other actors'/users' later consumption can still starve scheduling
            if self.nodes_per_task == 1:  # multi-node jobs are bounded by the PG readiness deadline instead
                outstanding, available = len(pending), ray.available_resources()
                for key, need in convert_remote_options_to_bundle(remote_options).items():
                    if need > (effective := available.get(key, 0) + outstanding * need):
                        job.error = RuntimeError(
                            f"requested {need} {key}/task but only {effective} available"
                            " — refusing to submit work that can never be scheduled"
                        )
                        job.retriable, job.reason = False, f"failed pre-submit ({job.error})"
                        failed.append(job)
                        return
            if self.nodes_per_task == 1:
                job.refs = tuple(remote_run.remote(executor_ref, r, i) for i, r in enumerate(job.ranks))
            else:
                self._launch_gang(executor_ref, job, remote_options)
            active.append(job)
            for ref in job.refs:
                pending[ref] = job

        def settle(job: RankJob) -> None:
            """Record a job's outcome once all its refs have settled; resubmit retriable failures within budget."""
            nonlocal completed
            active.remove(job)
            self._release_job(job)  # no-op on single node; frees gang actors + placement group for the next job
            if job.error is None:
                completed += 1
            elif not job.retriable or job.attempts > MAX_RESUBMITS:
                job.reason = f"failed after {job.attempts} attempt(s) ({'application error' if not job.retriable else 'retries exhausted'}): {job.error}"
                failed.append(job)
                logger.warning(f"Ranks {job.ranks} {job.reason}")
            else:
                logger.warning(f"ranks {job.ranks} failed ({job.error}); retrying (attempt {job.attempts + 1})")
                job.attempts += 1
                submit(job)
            while queue and len(active) < self.workers:
                submit(queue.popleft())

        try:
            while queue and len(active) < self.workers:
                submit(queue.popleft())

            while pending:
                # cancel jobs whose attempt overran its deadline; timed-out work never counts as done
                now = time.monotonic()
                for job in list(active):
                    if job.deadline is None or now < job.deadline or not any(r in pending for r in job.refs):
                        continue
                    for ref in job.refs:
                        pending.pop(ref, None)
                    self._release_job(job, job.refs)
                    job.error = TimeoutError(f"exceeded time limit of {self.time}s")
                    settle(job)
                # wait until the nearest deadline (bounded by 10s) or the next completion
                soonest = min((j.deadline for j in active if j.deadline is not None), default=None)
                poll = 10 if soonest is None else max(0.1, min(10, soonest - time.monotonic()))
                ready, _ = ray.wait(list(pending), num_returns=len(pending), timeout=poll)
                for ref in ready:
                    job = pending.pop(ref)
                    try:
                        ray.get(ref)
                    except Exception as e:  # ray-side infra errors are retriable; application errors are not
                        job.error, job.retriable = e, isinstance(e, _retriable_ray_errors())
                    if all(r not in pending for r in job.refs):
                        settle(job)

            if failed:
                details = "; ".join(f"ranks {job.ranks} {job.reason}" for job in failed)
                raise RuntimeError(f"{len(failed)}/{total} job(s) failed. {details}") from failed[0].error

            total_stats = PipelineStats()
            for rank in range(self.world_size):
                with self.logging_dir.open(f"stats/{rank:05d}.json", "r") as f:
                    total_stats += PipelineStats.from_json(json.load(f))
            with self.logging_dir.open("stats.json", "wt") as statsfile:
                total_stats.save_to_disk(statsfile)
            logger.success(total_stats.get_repr(f"All {completed}/{total} tasks."))
        finally:
            # kill whatever is still in flight, on every exit path (including KeyboardInterrupt, which propagates)
            for job in list(active):
                self._release_job(job, job.refs)

    def _run_for_rank(self, rank: int, local_rank: int = 0, node_rank: int = -1) -> PipelineStats:
        """Main executor's method: set up logging, pipe data through the pipeline steps, save stats, mark completion.
        node_rank 0 is the master (only one writing stats/markers), -1 = single node mode (default); local_rank is
        only used for logging (tasks with local_rank != 0 do not print logs to console).
        Args: rank: global rank to run; local_rank: rank within the job (only used for logging); node_rank: this
            node's index in a multi-node gang (0 = master), or -1 for single-node mode.
        Returns: the pipeline stats (empty if the rank was already completed; saved only when node_rank <= 0)."""
        import tempfile

        if self.is_rank_completed(rank):
            logger.info(f"Skipping {rank=} as it has already been completed.")
            return PipelineStats()

        # Set distributed environment variables
        self._set_distributed_environment(node_rank)

        # We log only locally to a unique per-job staging dir (node-local and shared, so colliding names would truncate) and upload to logging_dir after the pipeline is finished.
        staging_dir = get_datafolder(f"{tempfile.gettempdir()}/ray_logs/{uuid.uuid4().hex}")
        logfile = add_task_logger(staging_dir, rank, local_rank, node_rank=node_rank)
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

            # stats - only save on master node in distributed setting (or when node_rank <= 0 for single node)
            stats = PipelineStats(self.pipeline)
            if node_rank <= 0:
                with self.logging_dir.open(f"stats/{rank:05d}.json", "w") as f:
                    stats.save_to_disk(f)
                logger.info(stats.get_repr(f"Task {rank}"))
                # completed
                self.mark_rank_as_completed(rank)
        except Exception as e:
            logger.exception(e)
            raise e
        finally:
            close_task_logger(logfile)
            # Copy logs from local dir to logging_dir, then delete the staged copy (best effort)
            with (
                staging_dir.open(f"logs/task_{rank:05d}.log", "r") as f,
                self.logging_dir.open(f"logs/task_{rank:05d}.log", "w") as f_out,
            ):
                f_out.write(f.read())
            try:
                staging_dir.fs.rm(staging_dir.resolve_paths(""), recursive=True)
            except Exception as e:
                logger.warning(f"Failed to remove staging dir for {rank=}: {e}")
        return stats
