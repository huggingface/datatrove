#!/usr/bin/env python3
from collections import deque
import multiprocessing
import time
from typing import Callable, Optional, Sequence
from ray import cloudpickle

import ray

from datatrove.executor.base import PipelineExecutor
from datatrove.io import DataFolderLike
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.stats import PipelineStats
from datatrove.utils.logging import logger


@ray.remote
def run_for_rank(executor_pickled: bytes, ranks: list[int]) -> PipelineStats:
    """
        Main executor's method. Sets up logging, pipes data from each pipeline step to the next, saves statistics
        and marks tasks as completed.
    Args:
        rank: the rank that we want to run the pipeline for
        local_rank: at the moment this is only used for logging.
        Any task with local_rank != 0 will not print logs to console.

    Returns: the stats for this task

    """
    from datatrove.utils.stats import PipelineStats
    from datatrove.utils.logging import logger
    from ray import cloudpickle
    import multiprocess.pool

    executor = cloudpickle.loads(executor_pickled)
    rank_ids = list(range(len(ranks))) if executor.log_first else list(range(1, len(ranks) + 1))
    stats = PipelineStats()
    with multiprocess.pool.Pool(processes=len(ranks)) as pool:
        for task_result in pool.starmap(executor._run_for_rank, [(rank, rank_id) for rank_id, rank in zip(rank_ids, ranks)]):
            stats += task_result
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
        num_cpus_per_task: The number of CPUs to reserve per rank/task
            in the Ray cluster. Defaults to 1.
        memory_bytes_per_task: Amount of memory (in bytes) to reserve per rank/task
            in the Ray cluster. Defaults to 2 GB.
        num_gpus_per_task: The number of GPUs to reserve per rank/task in the
            Ray cluster. Defaults to 0.
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
        mem_per_cpu_gb: int = 2,
        ray_remote_kwargs: dict = None,
        log_first: bool = False,
        tasks_per_job: int = 1,
        timeout: Optional[int] = None,
    ):
        super().__init__(pipeline, logging_dir, skip_completed, randomize_start_duration)
        self.tasks = tasks
        self.workers = workers if workers != -1 else tasks
        self.depends = depends
        # track whether run() has been called
        self.mem_per_cpu_gb = mem_per_cpu_gb
        self.ray_remote_kwargs = ray_remote_kwargs
        self.tasks_per_job = tasks_per_job
        self.log_first = log_first
        self.timeout = timeout

    @property
    def world_size(self) -> int:
        return self.tasks

    def run(self):
        """
        Run the pipeline for each rank using Ray tasks.
        """

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

        # Cloudpickle the executor
        executor_pickled = cloudpickle.dumps(self)

        # 5) Define resource requirements for this pipeline's tasks
        remote_options = {
            "num_cpus": self.mem_per_cpu_gb,
            "num_gpus": 0,
            "memory": self.mem_per_cpu_gb * 1024 * 1024 * 1024,
        }
        if self.ray_remote_kwargs:
            remote_options.update(self.ray_remote_kwargs)

        # 6) Dispatch Ray tasks
        MAX_CONCURRENT_TASKS = self.workers
        ranks_per_jobs = [incomplete_ranks[i:i+self.tasks_per_job] for i in range(0, len(incomplete_ranks), self.tasks_per_job)]
        unfinished = []
        completed = 0

        # 7) Keep tasks start_time
        task_start_times = {}
        for _ in range(min(MAX_CONCURRENT_TASKS, len(ranks_per_jobs))):
            ranks_to_submit = ranks_per_jobs.pop()
            task = run_for_rank.options(**remote_options).remote(executor_pickled, ranks_to_submit)
            unfinished.append(task)
            task_start_times[task] = time.time()

        # 7) Wait for the tasks to finish, merging them as they complete.
        total_stats = PipelineStats()
        while unfinished:
            finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=10)
            for task in finished:
                del task_start_times[task]

            try:
                results = ray.get(finished)
                for task_result in results:
                    total_stats += task_result
                    completed += 1
            except Exception as e:
                logger.exception(f"Error processing shard: {e}")

            # If we have more shard paths left to process and we haven't hit the max
            # number of concurrent tasks, add tasks to the unfinished queue.
            while ranks_per_jobs and len(unfinished) < MAX_CONCURRENT_TASKS:
                ranks_to_submit = ranks_per_jobs.pop()
                task = run_for_rank.options(**remote_options).remote(executor_pickled, ranks_to_submit)
                unfinished.append(task)
                task_start_times[task] = time.time()

            # Finally remove tasks that run for more than self.timeout seconds
            if self.timeout:
                for task in unfinished:
                    if time.time() - task_start_times[task] > self.timeout:
                        del task_start_times[task]
                        unfinished.remove(task)
                        logger.warning(f"Task {task} timed out after {self.timeout} seconds and was removed from the queue.")
        logger.info("All Ray tasks have finished.")

        # 8) Save merged stats
        with self.logging_dir.open("stats.json", "wt") as statsfile:
            total_stats.save_to_disk(statsfile)

        if completed > 0:
            logger.success(total_stats.get_repr(f"All {completed}/{self.world_size} tasks"))
        return total_stats