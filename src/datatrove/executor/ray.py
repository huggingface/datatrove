#!/usr/bin/env python3
import json
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ray.exceptions import TaskCancelledError
from ray.util.client import ray
from ray.util.placement_group import PlacementGroup

from datatrove.executor.base import PipelineExecutor
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.logging import add_task_logger, close_task_logger, log_pipeline, logger
from datatrove.utils.stats import PipelineStats


@dataclass
class PlacementGroupTaskFuture:
    """Promise for a task in a placement group."""

    def __init__(self):
        self.tasks: list[ray.ObjectRef] | None = None

    def get_no_wait(self) -> list[ray.ObjectRef] | None:
        return self.tasks


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
        os.environ["RAY_NODELIST"] = ",".join(node_ips)
        os.environ["RAY_NODEID"] = str(node_idx)

        # Sleep for the executor's timeout
        def run_for_rank_wrapper_with_sleep(rank, rank_id):
            time.sleep(random.randint(0, executor_ref.randomize_start_duration))
            return executor_ref._run_for_rank(rank, rank_id)

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


def run_for_rank(executor_ref: "RayPipelineExecutor", ranks: list[int]) -> PipelineStats:
    """
        Main executor's method. Sets up logging, pipes data from each pipeline step to the next, saves statistics
        and marks tasks as completed.
    Args:
        executor_ref: the executor reference
        ranks: the ranks that we want to run in this job
    Returns: cumulative stats for all ranks in this job
    """
    import multiprocess.pool

    from datatrove.utils.stats import PipelineStats

    # Sleep for the executor's timeout
    def run_for_rank_wrapper_with_sleep(rank, rank_id):
        time.sleep(random.randint(0, executor_ref.randomize_start_duration))
        return executor_ref._run_for_rank(rank, rank_id)

    executor = executor_ref
    rank_ids = list(range(len(ranks))) if executor.log_first else list(range(1, len(ranks) + 1))
    stats = PipelineStats()
    # We use simple map, so that all tasks are executed and errors are reported (raised) only after all tasks are finished
    with multiprocess.pool.Pool(processes=len(ranks)) as pool:
        # Consume results
        deque(
            pool.starmap(run_for_rank_wrapper_with_sleep, [(rank, rank_id) for rank_id, rank in zip(rank_ids, ranks)]),
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
    tasks: list[ray.ObjectRef]
    workers: list[ray.ActorHandle]
    placement_group: PlacementGroup
    tasks_successfully_completed: int = 0


class TimeoutManager:
    """
    Manages timeouts for tasks.
    """

    def __init__(self, timeout_seconds: int | None = None):
        self.task_start_times = {}
        self.timeout_seconds = timeout_seconds

    def add_task(self, task: ray.ObjectRef):
        self.task_start_times[task] = time.time()

    def remove_task(self, task: ray.ObjectRef):
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
    ) -> PlacementGroupTaskFuture:
        pg_task_future = PlacementGroupTaskFuture()

        def async_submit_task():
            pg = ray.util.placement_group(
                [convert_remote_options_to_bundle(remote_options)] * self.nodes_per_task, strategy="STRICT_SPREAD"
            )
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

            group = TaskGroup(tasks=tasks, workers=workers, placement_group=pg)
            for task in tasks:
                self.task_to_group[task] = group
                self.timeout_manager.add_task(task)

            pg_task_future.tasks = tasks

        # Submit to thread pool executor and store the future
        thread_future = self.executor.submit(async_submit_task)
        self.pg_futures.append(thread_future)
        return pg_task_future

    def wait(
        self, tasks: list[ray.ObjectRef | PlacementGroupTaskFuture], timeout: int = 0, num_returns: int = 1
    ) -> tuple[list[ray.ObjectRef], list[ray.ObjectRef | PlacementGroupTaskFuture]]:
        """
        Functions first checks for completions of the placmenetgroup tasks. Then it takes those that are finished, adds the to normal ray objects calls wait on that.
        Finally it returns the list of finished tasks and the (list of tasks that are still running + placement group tasks that hasven't finished yet)
        """
        # Separate placement group tasks from regular ray tasks
        placement_group_tasks = [task for task in tasks if isinstance(task, PlacementGroupTaskFuture)]
        ray_tasks = [task for task in tasks if not isinstance(task, PlacementGroupTaskFuture)]

        unfinished_pg_tasks = []
        for pg_task in placement_group_tasks:
            if pg_task.tasks is not None:
                ray_tasks.extend(pg_task.tasks)
            else:
                unfinished_pg_tasks.append(pg_task)

        finished, unfinished = ray.wait(ray_tasks, num_returns=num_returns, timeout=timeout)
        return finished, unfinished + unfinished_pg_tasks

    def task_done(self, task: ray.ObjectRef) -> bool:
        """
        Marks task as done and potentially cleans up the placement group.
        Returns True if all tasks in the group are done and were completed successfully (no errors).

        Note: This is the only task which can manipulate the task groups as well as delete placement groups.
        """
        try:
            # Hope this is cheap since we know the task is done :pray:
            ray.get(task)
            self.task_to_group[task].tasks_successfully_completed += 1
        except Exception:
            pass

        # Delete the task from group
        group = self.task_to_group[task]
        group.tasks.remove(task)
        # Remove the reference to group
        del self.task_to_group[task]
        self.timeout_manager.remove_task(task)

        if len(group.tasks) == 0:
            # Remove the placement group
            try:
                ray.util.remove_placement_group(group.placement_group)
            except Exception as e:
                logger.warning(f"Failed to remove placement group: {e}")

        return group.tasks_successfully_completed == self.nodes_per_task

    def kill_task_group(self, task: ray.ObjectRef):
        """
        Kills a task and its group siblings
        """
        if task not in self.task_to_group:
            logger.warning(f"Task {task} not found in task manager")
            return

        group = self.task_to_group[task]

        # Kill all tasks in the group
        for t in group.tasks:
            try:
                ray.kill(t, force=True)
            except TaskCancelledError:
                # Task was already cancelled, so we can ignore it
                pass

    def clean_up(self):
        """
        Cleans up the task manager.
        """
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
        ray_remote_kwargs: dict = None,
        log_first: bool = False,
        tasks_per_job: int = 1,
        time: Optional[int] = None,
        nodes_per_task: int = 1,
    ):
        super().__init__(pipeline, logging_dir, skip_completed, randomize_start_duration)
        self.tasks = tasks
        self.workers = workers if workers != -1 else tasks
        self.depends = depends
        # track whether run() has been called
        self.cpus_per_task = cpus_per_task
        self.mem_per_cpu_gb = mem_per_cpu_gb
        self.ray_remote_kwargs = ray_remote_kwargs
        self.tasks_per_job = tasks_per_job
        self.log_first = log_first
        self.time = time
        self.nodes_per_task = nodes_per_task

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

        # 1) If there is a depends=, ensure it has run and is finished
        if self.depends:
            logger.info(f'Launching dependency job "{self.depends}"')
            self.depends.run()

        # 2) Check if all tasks are already completed
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
            "num_gpus": 0,
            "memory": int(self.mem_per_cpu_gb * self.cpus_per_task * 1024 * 1024 * 1024),
        }
        if self.ray_remote_kwargs:
            remote_options.update(self.ray_remote_kwargs)

        # 5) Prepare ranks per job
        MAX_CONCURRENT_TASKS = self.workers
        ranks_per_jobs = [
            incomplete_ranks[i : i + self.tasks_per_job] for i in range(0, len(incomplete_ranks), self.tasks_per_job)
        ]
        unfinished = []
        total_tasks = len(ranks_per_jobs)
        completed = 0

        # Initialize task group manager
        timeout_manager = TimeoutManager(self.time)
        task_manager = RayTaskManager(self.nodes_per_task, timeout_manager)
        finished, unfinished = [], []
        # Launch initial tasks

        try:
            for _ in range(min(MAX_CONCURRENT_TASKS, len(ranks_per_jobs))):
                ranks_to_submit = ranks_per_jobs.pop(0)
                pg_task_future = task_manager.submit_task(executor_ref, ranks_to_submit, remote_options)
                unfinished.append(pg_task_future)

            # Wait for tasks to finish
            while unfinished:
                finished, unfinished = task_manager.wait(unfinished, num_returns=1, timeout=10)

                # Handle timeouts (and completed tasks)
                for task in finished:
                    if task_manager.task_done(task):
                        completed += 1

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

    def _run_for_rank(self, rank: int, local_rank: int = 0) -> PipelineStats:
        """
            Main executor's method. Sets up logging, pipes data from each pipeline step to the next, saves statistics
            and marks tasks as completed.
        Args:
            rank: the rank that we want to run the pipeline for
            local_rank: at the moment this is only used for logging.
            Any task with local_rank != 0 will not print logs to console.

        Returns: the stats for this task
        """
        import tempfile

        if self.is_rank_completed(rank):
            logger.info(f"Skipping {rank=} as it has already been completed.")
            return PipelineStats()

        # We log only locally and upload logs to logging_dir after the pipeline is finished
        ray_logs_dir = get_datafolder(f"{tempfile.gettempdir()}/ray_logs")
        logfile = add_task_logger(ray_logs_dir, rank, local_rank)
        log_pipeline(self.pipeline)

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

            # stats
            stats = PipelineStats(self.pipeline)
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
