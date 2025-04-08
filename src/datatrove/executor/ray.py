#!/usr/bin/env python3
from collections import deque
import multiprocessing
import random
import time
from typing import Callable, Optional, Sequence

import ray

from datatrove.executor.base import PipelineExecutor
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.utils.stats import PipelineStats
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import add_task_logger, close_task_logger, log_pipeline, logger


@ray.remote
def run_for_rank(executor_ref: "RayPipelineExecutor", ranks: list[int]) -> PipelineStats:
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
    import multiprocess.pool

    executor = executor_ref
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
        ranks_per_jobs = [incomplete_ranks[i:i+self.tasks_per_job] for i in range(0, len(incomplete_ranks), self.tasks_per_job)]
        unfinished = []
        completed = 0

        ray_remote_func = run_for_rank.options(**remote_options)

        # 7) Keep tasks start_time
        task_start_times = {}
        for _ in range(min(MAX_CONCURRENT_TASKS, len(ranks_per_jobs))):
            ranks_to_submit = ranks_per_jobs.pop()
            task = ray_remote_func.remote(executor_ref, ranks_to_submit)
            unfinished.append(task)
            task_start_times[task] = time.time()

        # 7) Wait for the tasks to finish, merging them as they complete.
        total_stats = PipelineStats()
        while unfinished:
            finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=10)
            for task in finished:
                # Remove task from task_start_times
                del task_start_times[task]
                # Remove task itself
                del task

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
                task = ray_remote_func.remote(executor_ref, ranks_to_submit)
                unfinished.append(task)
                task_start_times[task] = time.time()

            # Finally remove tasks that run for more than self.timeout seconds
            if self.time:
                for task in unfinished:
                    if time.time() - task_start_times[task] > self.time:
                        del task_start_times[task]
                        unfinished.remove(task)
                        logger.warning(f"Task {task} timed out after {self.time} seconds and was removed from the queue.")
        logger.info("All Ray tasks have finished.")

        # 8) Save merged stats
        with self.logging_dir.open("stats.json", "wt") as statsfile:
            total_stats.save_to_disk(statsfile)

        if completed > 0:
            logger.success(total_stats.get_repr(f"All {completed}/{self.world_size} tasks"))
        return total_stats

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
        if self.is_rank_completed(rank):
            logger.info(f"Skipping {rank=} as it has already been completed.")
            return PipelineStats()
        
        # We log only locally and upload logs to s3 after the pipeline is finished
        logfile = add_task_logger(get_datafolder("/tmp/ray_logs"), rank, local_rank)
        log_pipeline(self.pipeline)

        if self.randomize_start_duration > 0:
            time.sleep(random.randint(0, self.randomize_start_duration))
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
            # Upload logs to s3
            with open(logfile, "rt") as f, self.logging_dir.open(f"logs/{rank:05d}.log", "wt") as f_out:
                f_out.write(f.read())
        return stats