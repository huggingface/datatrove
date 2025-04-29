#!/usr/bin/env python3
import json
import random
import time
from collections import deque
from typing import Callable, Optional, Sequence

from datatrove.executor.base import PipelineExecutor
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.logging import add_task_logger, close_task_logger, log_pipeline, logger
from datatrove.utils.stats import PipelineStats


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


class RayPipelineExecutor(PipelineExecutor):
    """
    Executor to run a pipeline using Ray. It's expected that the Ray cluster has already
    been set up (e.g., via `ray.init()`) prior to invoking this pipeline.

    Args:
        pipeline: a list of PipelineStep and/or custom lamdba functions
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

    @property
    def world_size(self) -> int:
        return self.tasks

    def run(self):
        """
        Run the pipeline for each rank using Ray tasks.
        """

        check_required_dependencies("ray", ["ray"])
        import ray

        # 1) If there is a depends=, ensure it has run and is finished
        if self.depends:
            logger.info(f'Launching dependency job "{self.depends}"')
            self.depends.run()

        # 3) Check if all tasks are already completed
        incomplete_ranks = self.get_incomplete_ranks(range(self.world_size))
        if not incomplete_ranks:
            logger.info(f"All {self.world_size} tasks appear to be completed already. Nothing to run.")
            return

        logger.info(f"Will run pipeline on {len(incomplete_ranks)} incomplete ranks out of {self.world_size} total.")

        # 4) Save executor JSON
        self.save_executor_as_json()

        executor_ref = ray.put(self)

        # 5) Define resource requirements for this pipeline's tasks
        remote_options = {
            "num_cpus": self.cpus_per_task,
            "num_gpus": 0,
            "memory": int(self.mem_per_cpu_gb * self.cpus_per_task * 1024 * 1024 * 1024),
        }
        if self.ray_remote_kwargs:
            remote_options.update(self.ray_remote_kwargs)

        # 6) Dispatch Ray tasks
        MAX_CONCURRENT_TASKS = self.workers
        ranks_per_jobs = [
            incomplete_ranks[i : i + self.tasks_per_job] for i in range(0, len(incomplete_ranks), self.tasks_per_job)
        ]
        unfinished = []
        total_tasks = len(ranks_per_jobs)
        completed = 0

        ray_remote_func = ray.remote(**remote_options)(run_for_rank)

        # 7) Keep tasks start_time
        task_start_times = {}
        for _ in range(min(MAX_CONCURRENT_TASKS, len(ranks_per_jobs))):
            ranks_to_submit = ranks_per_jobs.pop(0)
            task = ray_remote_func.remote(executor_ref, ranks_to_submit)
            unfinished.append(task)
            task_start_times[task] = time.time()

        # 7) Wait for the tasks to finish, merging them as they complete.
        while unfinished:
            finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=10)
            for task in finished:
                # Remove task from task_start_times
                del task_start_times[task]
                # Remove task itself
                del task

            try:
                results = ray.get(finished)
                for _ in results:
                    completed += 1
            except Exception as e:
                logger.exception(f"Error processing rank: {e}")

            # If we have more ranks left to process and we haven't hit the max
            # number of concurrent tasks, add tasks to the unfinished queue.
            while ranks_per_jobs and len(unfinished) < MAX_CONCURRENT_TASKS:
                ranks_to_submit = ranks_per_jobs.pop(0)
                task = ray_remote_func.remote(executor_ref, ranks_to_submit)
                unfinished.append(task)
                task_start_times[task] = time.time()

            # Finally remove tasks that run for more than self.timeout seconds
            if self.time:
                for task in unfinished:
                    if time.time() - task_start_times[task] > self.time:
                        # No mercy :) -> should call SIGKILL
                        ray.kill(task, force=True)
                        del task_start_times[task]
                        unfinished.remove(task)
                        logger.warning(f"Task {task} timed out after {self.time} seconds and was killed.")
        logger.info("All Ray tasks have finished.")

        # 8) Merge stats of all ranks
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
