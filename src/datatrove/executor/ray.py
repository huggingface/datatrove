#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from datatrove.executor.base import DistributedEnvVars, PipelineExecutor
from datatrove.io import DataFolderLike, file_is_local, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.logging import add_task_logger, close_task_logger, log_pipeline, logger
from datatrove.utils.stats import PipelineStats


if TYPE_CHECKING:
    from ray import ObjectRef
    from ray.actor import ActorHandle
    from ray.util.placement_group import PlacementGroup

# TODO: We should re-use the placement group from the previous run if it exists


class RankWorker:
    """Ray actor for running pipeline tasks with multi-node support."""

    def get_node_ip(self) -> str:
        import ray

        """Get the IP address of the node this actor is running on."""
        return ray.util.get_node_ip_address()

    def run_for_rank(
        self,
        executor_ref: "RayPipelineExecutor",
        ranks: list[int],
        node_ips: list[str],
        node_idx: int,
    ) -> PipelineStats:
        """
        Main executor's method with multi-node support. Sets up logging, pipes data from each pipeline step
        to the next, saves statistics and marks tasks as completed.

        Args:
            executor_ref: the executor reference
            ranks: the ranks that we want to run in this job
            node_ips: list of IP addresses for all nodes in the placement group
            node_idx: index of this node in the node_ips list (0 = master)
        Returns: cumulative stats for all ranks in this job
        """
        import os

        import multiprocess.pool

        # Set environment variables for distributed execution
        # Sleep for the executor's timeout
        def run_for_rank_wrapper_with_sleep(rank, rank_id):
            os.environ["RAY_NODELIST"] = ",".join(node_ips)
            os.environ["RAY_NODEID"] = str(node_idx)
            time.sleep(random.randint(0, executor_ref.randomize_start_duration))
            # Use -1 for single-node mode, otherwise use node_idx
            node_rank = -1 if executor_ref.nodes_per_task == 1 else node_idx
            return executor_ref._run_for_rank(rank, rank_id, node_rank=node_rank)

        executor = executor_ref
        rank_ids = list(range(len(ranks))) if executor.log_first else list(range(1, len(ranks) + 1))
        stats = PipelineStats()
        # We use simple map, so that all tasks are executed and errors are reported (raised) only after all tasks are finished
        with multiprocess.pool.Pool(processes=len(ranks)) as pool:
            # Consume results
            deque(
                pool.starmap(
                    run_for_rank_wrapper_with_sleep,
                    [(rank, rank_id) for rank_id, rank in zip(rank_ids, ranks)],
                ),
                maxlen=0,
            )
        return stats


def convert_remote_options_to_bundle(remote_options: dict) -> dict:
    """
    Converts remote options to a bundle for a placement group.
    """
    bundle = {}
    if "num_cpus" in remote_options:
        bundle["CPU"] = remote_options["num_cpus"]
    if "memory" in remote_options:
        bundle["memory"] = remote_options["memory"]
    if "num_gpus" in remote_options:
        bundle["GPU"] = remote_options["num_gpus"]
    return bundle


@dataclass
class TaskGroup:
    tasks: list["ObjectRef"]
    workers: list["ActorHandle"]
    placement_group: "PlacementGroup"
    ranks: list[int]  # Ranks being processed by this task group
    tasks_successfully_completed: int = 0
    has_retriable_error: bool = False  # Track if any task in group failed with retriable error


class TimeoutManager:
    """
    Manages timeouts for tasks.
    """

    def __init__(self, timeout_seconds: int | None = None):
        self.task_start_times = {}
        self.timeout_seconds = timeout_seconds

    def add_task(self, task: "ObjectRef"):
        self.task_start_times[task] = time.time()

    def remove_task(self, task: "ObjectRef"):
        del self.task_start_times[task]

    def check_timeouts(self):
        """
        Checks if any tasks have timed out. Returns a list of tasks that have timed out.
        """
        if self.timeout_seconds is None:
            return []
        timed_out_tasks = []
        for task in self.task_start_times:
            if time.time() - self.task_start_times[task] > self.timeout_seconds:
                timed_out_tasks.append(task)

        return timed_out_tasks


class RayTaskManager:
    """
    Manages task groups for multi-node execution.
    Tracks tasks, their groups, and placement groups for coordinated execution.
    """

    def __init__(self, nodes_per_task: int, timeout_manager: TimeoutManager):
        """
        Initialize the task group manager.

        Args:
            nodes_per_task: Number of nodes per task group. If 1, no grouping is used.
        """
        self.nodes_per_task = nodes_per_task
        self.timeout_manager = timeout_manager
        self.task_to_group = {}
        self.pg_futures = []
        self.executor = ThreadPoolExecutor(max_workers=10)

    def submit_task(
        self, executor_ref: "RayPipelineExecutor", ranks_to_submit: list[int], remote_options: dict
    ) -> Future:
        import ray

        pg_future = Future()

        def async_submit_task():
            # No enforcment of "spread" here, thus it can happen that multiple tasks run on same nodes under nodes>=2.
            # It shouldn't be an issue, but we should be aware of it.
            pg = ray.util.placement_group([convert_remote_options_to_bundle(remote_options)] * self.nodes_per_task)
            ray.get(pg.ready())
            workers = []
            for i in range(self.nodes_per_task):
                worker = (
                    ray.remote(RankWorker)
                    .options(
                        placement_group=pg,
                        placement_group_bundle_index=i,
                        **remote_options,
                    )
                    .remote()
                )
                workers.append(worker)

            # Discover IPs
            ip_futures = [worker.get_node_ip.remote() for worker in workers]
            node_ips = ray.get(ip_futures)

            # Launch tasks (each processes ranks independently)
            tasks = []
            for node_idx, worker in enumerate(workers):
                task = worker.run_for_rank.remote(executor_ref, ranks_to_submit, node_ips, node_idx)
                tasks.append(task)

            group = TaskGroup(tasks=tasks, workers=workers, placement_group=pg, ranks=ranks_to_submit)
            for task in tasks:
                self.task_to_group[task] = group
                self.timeout_manager.add_task(task)

            pg_future.set_result(tasks)

        # Submit to thread pool executor
        thread_future = self.executor.submit(async_submit_task)

        self.pg_futures.append(thread_future)
        return pg_future

    def wait(
        self, tasks: list["ObjectRef | Future"], timeout: int = 0, num_returns: int = 1
    ) -> tuple[list["ObjectRef"], list["ObjectRef | Future"]]:
        """
        Functions first checks for completions of the placement group tasks. Then it takes those that are finished, adds them to normal ray objects and calls wait on that.
        Finally it returns the list of finished tasks and the (list of tasks that are still running + placement group tasks that haven't finished yet)
        """
        import ray

        # Separate placement group futures from regular ray tasks
        placement_group_futures = [task for task in tasks if isinstance(task, Future)]
        ray_tasks = [task for task in tasks if not isinstance(task, Future)]

        unfinished_pg_tasks = []
        if not ray_tasks and placement_group_futures:
            # All are placement group futures
            wait(placement_group_futures, timeout=timeout, return_when=FIRST_COMPLETED)

        for pg_future in placement_group_futures:
            # Check if the future is done and has tasks attribute
            if pg_future.done():
                try:
                    ray_tasks.extend(pg_future.result())
                except Exception as e:
                    # TODO: Handle this better, by knowing the rank ids
                    logger.warning(f"Failed to get result from placement group future: {e}")
            else:
                unfinished_pg_tasks.append(pg_future)

        if ray_tasks:
            finished, unfinished = ray.wait(ray_tasks, num_returns=num_returns, timeout=timeout)
            return finished, unfinished + unfinished_pg_tasks
        return [], unfinished_pg_tasks

    def task_done(self, task: "ObjectRef") -> tuple[bool, list[int] | None]:
        """
        Marks task as done and potentially cleans up the placement group.
        Returns a tuple:
            - (True, None): All tasks in the group are done and were completed successfully
            - (False, ranks_to_resubmit): Task failed due to retriable error, returns ranks to resubmit
            - (False, None): Task failed but not retriable, or group not fully done

        Note: This is the only function which can manipulate the task groups as well as delete placement groups.
        """
        import ray
        from ray.exceptions import (
            ObjectLostError,
            RayActorError,
            TaskCancelledError,
            WorkerCrashedError,
        )

        group = self.task_to_group[task]

        try:
            # Hope this is cheap since we know the task is done :pray:
            ray.get(task)
            group.tasks_successfully_completed += 1
        except (WorkerCrashedError, TaskCancelledError, ObjectLostError) as e:
            # These are retriable Ray-side errors
            group.has_retriable_error = True
            logger.warning(f"Task {task} failed with retriable error {type(e).__name__}: {e}")
        except RayActorError as e:
            # Check if it's a preemption (actor died)
            if hasattr(e, "preempted") and e.preempted:
                group.has_retriable_error = True
                logger.warning(f"Task {task} failed due to preemption: {e}")
            else:
                # Other actor errors might be retriable too (e.g., actor crashed)
                group.has_retriable_error = True
                logger.warning(f"Task {task} failed with RayActorError: {e}")
        except Exception as e:
            # Application-level errors are not retriable
            logger.debug(f"Task {task} failed with non-retriable error {type(e).__name__}: {e}")

        # Delete the task from group
        group.tasks.remove(task)
        # Remove the reference to group
        del self.task_to_group[task]
        self.timeout_manager.remove_task(task)

        if len(group.tasks) == 0:
            # All tasks in the group are done
            # Remove the placement group
            try:
                ray.util.remove_placement_group(group.placement_group)
            except Exception as e:
                logger.warning(f"Failed to remove placement group: {e}")

            # Return success status and ranks to resubmit if needed
            if group.tasks_successfully_completed == self.nodes_per_task:
                return True, None
            elif group.has_retriable_error:
                # Return ranks to resubmit if any task failed with retriable error
                return False, group.ranks
            else:
                return False, None

        # Group not fully done yet
        return False, None

    def kill_task_group(self, task: "ObjectRef"):
        """
        Kills a task and its group siblings
        """
        import ray
        from ray.exceptions import TaskCancelledError

        if task not in self.task_to_group:
            logger.warning(f"Task {task} not found in task manager")
            return

        group = self.task_to_group[task]

        # Kill all tasks in the group
        for t in group.tasks:
            try:
                ray.cancel(t)
            except TaskCancelledError:
                # Task was already cancelled, so we can ignore it
                pass

        # Wait for all tasks to be cancelled, with timeout
        ray.wait(group.tasks, num_returns=len(group.tasks), timeout=2)

        # Kill actors
        for worker in group.workers:
            try:
                ray.kill(worker)
            except Exception as e:
                logger.warning(f"Failed to kill worker: {e}")

        # Wait for all tasks to be cancelled (due to worker kill), with timeout
        ray.wait(group.tasks, num_returns=len(group.tasks), timeout=2)

        # Remove placement group
        try:
            ray.util.remove_placement_group(group.placement_group)
        except Exception as e:
            logger.warning(f"Failed to remove placement group: {e}")

        # Clean up references
        for t in list(group.tasks):
            if t in self.task_to_group:
                del self.task_to_group[t]
            if t in self.timeout_manager.task_start_times:
                self.timeout_manager.remove_task(t)

    def clean_up(self):
        """
        Cleans up the task manager, invalidates any placement groups and tasks that are still running.
        """
        import ray

        # First cancel all placement group futures (ThreadPoolExecutor futures)
        for future in self.pg_futures:
            future.cancel()

        # Kill all remaining task groups
        for task in list(self.task_to_group.keys()):
            self.kill_task_group(task)

        # Wait for any remaining tasks to finish with timeout
        for task in list(self.task_to_group.keys()):
            try:
                ray.get(task)
            except Exception:
                pass

        # Shutdown the thread pool executor
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clean_up()


class RayPipelineExecutor(PipelineExecutor):
    """
    Executor to run a pipeline using Ray. It's expected that the Ray cluster has already
    been set up (e.g., via `ray.init()`) prior to invoking this pipeline.

    Args:
        pipeline: a list of PipelineStep and/or custom lambda functions
            with arguments (data: DocumentsPipeline, rank: int,
            world_size: int)
        tasks: total number of tasks to run the pipeline on (default: 1)
        workers: how many tasks to run simultaneously. (default is -1 for no limit aka tasks)
        depends: another PipelineExecutor that should run before this one
        skip_completed: whether to skip tasks that were completed in
            previous runs. default: True
        logging_dir: where to save logs, stats, etc. Should be parsable into a datatrove.io.DataFolder
        randomize_start_duration: the maximum number of seconds to delay the start of each task.
        cpus_per_task: The number of CPUs to reserve per task
            in the Ray cluster. Defaults to 1.
        mem_per_cpu_gb: Amount of memory (in GB) to reserve per CPU
            in the Ray cluster. Defaults to 2 GB.
        gpus_per_task: The number of GPUs to reserve per task
        nodes_per_task: Number of nodes to use per task. If > 1, creates a placement group
        ray_remote_kwargs: Additional kwargs to pass to the ray.remote decorator
        log_first: Whether to the first task in ray job should log to console. Default: False
        tasks_per_job: Number of tasks to run in each Ray job. Default: 1
        time: Optional time limit in seconds for each task
        nodes_per_task: Number of nodes to use per task. If > 1, creates a placement group
            and launches one task per node. Each task processes the same ranks independently.
            Default: 1 (single node per task)
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
        time: Optional[int] = None,
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
        self.log_first = log_first
        self.time = time
        self._launched = False
        self.nodes_per_task = nodes_per_task

    def get_distributed_env(self, node_rank: int = -1) -> DistributedEnvVars:
        """Get distributed environment variables for RAY executor."""
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
        return self.tasks

    def run(self):
        """
        Run the pipeline for each rank using Ray tasks.
        Supports both single-node (nodes_per_task=1) and multi-node (nodes_per_task>1) execution.
        """

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
            while (
                incomplete := len(self.depends.get_incomplete_ranks(skip_completed=True))
            ) > 0:  # set skip_completed=True to get *real* incomplete task count
                logger.info(f"Dependency job still has {incomplete}/{self.depends.world_size} tasks. Waiting...")
                time.sleep(2 * 60)

        self._launched = True
        # 3) Check if all tasks are already completed
        incomplete_ranks = self.get_incomplete_ranks(range(self.world_size))
        if not incomplete_ranks:
            logger.info(f"All {self.world_size} tasks appear to be completed already. Nothing to run.")
            return

        logger.info(f"Will run pipeline on {len(incomplete_ranks)} incomplete ranks out of {self.world_size} total.")

        # 3) Save executor JSON
        self.save_executor_as_json()

        executor_ref = ray.put(self)

        # 4) Define resource requirements for this pipeline's tasks
        remote_options = {
            "num_cpus": self.cpus_per_task,
            "num_gpus": self.gpus_per_task,
            "memory": int(self.mem_per_cpu_gb * self.cpus_per_task * 1024 * 1024 * 1024),
        }
        if self.ray_remote_kwargs:
            remote_options.update(self.ray_remote_kwargs)

        # 5) Prepare ranks per job
        MAX_CONCURRENT_TASKS = self.workers
        ranks_per_jobs = [
            incomplete_ranks[i : i + self.tasks_per_job] for i in range(0, len(incomplete_ranks), self.tasks_per_job)
        ]
        total_tasks = len(ranks_per_jobs)
        completed = 0

        # Initialize task group manager
        timeout_manager = TimeoutManager(self.time)
        task_manager = RayTaskManager(self.nodes_per_task, timeout_manager)
        finished, unfinished = [], []
        # Track resubmission counts to prevent infinite loops
        rank_resubmit_count: dict[tuple[int, ...], int] = {}
        max_resubmits = 3  # Maximum number of resubmissions per rank

        # Launch initial tasks
        try:
            for _ in range(min(MAX_CONCURRENT_TASKS, len(ranks_per_jobs))):
                ranks_to_submit = ranks_per_jobs.pop(0)
                pg_task_future = task_manager.submit_task(executor_ref, ranks_to_submit, remote_options)
                unfinished.append(pg_task_future)

            # Wait for tasks to finish
            while unfinished:
                finished, unfinished = task_manager.wait(unfinished, num_returns=1, timeout=10)

                # Handle completed tasks and resubmit if needed
                for task in finished:
                    success, ranks_to_resubmit = task_manager.task_done(task)
                    if success:
                        completed += 1

                    if ranks_to_resubmit:
                        ranks_to_resubmit = tuple(ranks_to_resubmit)
                        resubmit_rank_count = rank_resubmit_count.get(ranks_to_resubmit, 0)
                        if resubmit_rank_count >= max_resubmits:
                            logger.warning(
                                f"Rank {ranks_to_resubmit} has failed too many times, skipping further resubmissions"
                            )
                            continue
                        rank_resubmit_count[ranks_to_resubmit] = resubmit_rank_count + 1
                        ranks_per_jobs.append(ranks_to_resubmit)

                for task in timeout_manager.check_timeouts():
                    task_manager.kill_task_group(task)

                # Launch new tasks if slots available
                while ranks_per_jobs and len(unfinished) < MAX_CONCURRENT_TASKS:
                    ranks_to_submit = ranks_per_jobs.pop(0)
                    pg_task_future = task_manager.submit_task(executor_ref, ranks_to_submit, remote_options)
                    unfinished.append(pg_task_future)

            logger.info("All Ray tasks have finished.")

            # 6) Merge stats of all ranks
            if completed == total_tasks:
                total_stats = PipelineStats()
                for rank in range(self.world_size):
                    with self.logging_dir.open(f"stats/{rank:05d}.json", "r") as f:
                        total_stats += PipelineStats.from_json(json.load(f))
                with self.logging_dir.open("stats.json", "wt") as statsfile:
                    total_stats.save_to_disk(statsfile)
                logger.success(total_stats.get_repr(f"All {completed}/{total_tasks} tasks."))
            else:
                logger.warning(f"Only {completed}/{total_tasks} tasks completed.")
        finally:
            # Clean up task manager (shuts down thread pool executor)
            task_manager.clean_up()

    def _run_for_rank(self, rank: int, local_rank: int = 0, node_rank: int = -1) -> PipelineStats:
        """
            Main executor's method. Sets up logging, pipes data from each pipeline step to the next, saves statistics
            and marks tasks as completed.
        Args:
            rank: the rank that we want to run the pipeline for
            local_rank: at the moment this is only used for logging.
            node_rank: node rank/ID for logging prefix. Logs will be prefixed with [NODE X] if node_rank != -1. We assume node_rank == 0 is the master node. -1 means single node mode (default).
            Any task with local_rank != 0 will not print logs to console.

        Returns: the stats for this task
        """
        import tempfile

        if self.is_rank_completed(rank):
            logger.info(f"Skipping {rank=} as it has already been completed.")
            return PipelineStats()

        # Set distributed environment variables
        self._set_distributed_environment(node_rank)

        # We log only locally and upload logs to logging_dir after the pipeline is finished
        ray_logs_dir = get_datafolder(f"{tempfile.gettempdir()}/ray_logs")
        logfile = add_task_logger(ray_logs_dir, rank, local_rank, node_rank=node_rank)
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
            # Copy logs from local dir to logging_dir
            with (
                ray_logs_dir.open(f"logs/task_{rank:05d}.log", "r") as f,
                self.logging_dir.open(f"logs/task_{rank:05d}.log", "w") as f_out,
            ):
                f_out.write(f.read())
        return stats
