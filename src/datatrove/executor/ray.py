#!/usr/bin/env python3
import time
from typing import Callable
import ray

from datatrove.executor.base import PipelineExecutor
from datatrove.io import DataFolderLike
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from datatrove.utils.stats import PipelineStats


@ray.remote
def _launch_run_for_rank(executor_pik, rank: int) -> PipelineStats:
    """
    A Ray remote function that invokes executor_pik._run_for_rank(rank).
    We define it at the module level so Ray can easily serialize/deserialize it.
    """
    # Note: 'executor_pik' is expected to be an actual executor instance if we pass 'self',
    # or a copy (e.g., via deepcopy) that is Ray-picklable.
    return executor_pik._run_for_rank(rank)


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
        num_cpus_per_task: float = 1,
        memory_bytes_per_task: int = 2 * 1024 * 1024 * 1024,
        num_gpus_per_task: float = 0,
    ):
        super().__init__(pipeline, logging_dir, skip_completed, randomize_start_duration)
        self.tasks = tasks
        self.workers = workers if workers != -1 else tasks
        self.depends = depends
        # track whether run() has been called
        self._launched = False
        self.num_cpus_per_task = num_cpus_per_task
        self.memory_bytes_per_task = memory_bytes_per_task
        self.num_gpus_per_task = num_gpus_per_task

    @property
    def world_size(self) -> int:
        return self.tasks

    def run(self):
        """
        Run the pipeline for each rank using Ray tasks.
        """

        # 1) If there is a depends=, ensure it has run and is finished
        if self.depends:
            if not self.depends._launched:
                logger.info(f'Launching dependency job "{self.depends}"')
                self.depends.run()

            # Wait until the dependency has no incomplete ranks
            while True:
                incomplete = len(self.depends.get_incomplete_ranks())
                if incomplete == 0:
                    break
                logger.info(f"Dependency job still has {incomplete}/{self.depends.world_size} tasks. Waiting...")
                time.sleep(2 * 60)

        # 2) Mark this executor as launched
        self._launched = True

        # 3) Check if all tasks are already completed
        incomplete_ranks = self.get_incomplete_ranks(range(self.world_size))
        if not incomplete_ranks:
            logger.info(f"All {self.world_size} tasks appear to be completed already. Nothing to run.")
            return

        logger.info(f"Will run pipeline on {len(incomplete_ranks)} incomplete ranks out of {self.world_size} total.")

        # 4) Save executor JSON
        self.save_executor_as_json()

        # 5) Define resource requirements for this pipeline's tasks
        remote_options = {
            "num_cpus": self.num_cpus_per_task,
            "num_gpus": self.num_gpus_per_task,
            "memory": self.memory_bytes_per_task,
        }

        # 6) Dispatch Ray tasks
        MAX_CONCURRENT_TASKS = self.workers
        unfinished = []

        for _ in range(min(MAX_CONCURRENT_TASKS, len(incomplete_ranks))):
            rank_to_submit = incomplete_ranks.pop()
            unfinished.append(_launch_run_for_rank.options(**remote_options).remote(self, rank_to_submit))

        # 7) Wait for the tasks to finish, merging them as they complete.
        total_stats = PipelineStats()
        while unfinished:
            finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
            try:
                results = ray.get(finished)
                for task_result in results:
                    total_stats += task_result
            except Exception as e:
                logger.exception(f"Error processing shard: {e}")
                raise

            # If we have more shard paths left to process and we haven't hit the max
            # number of concurrent tasks, add tasks to the unfinished queue.
            while incomplete_ranks and len(unfinished) < MAX_CONCURRENT_TASKS:
                rank_to_submit = incomplete_ranks.pop()
                unfinished.append(_launch_run_for_rank.options(**remote_options).remote(self, rank_to_submit))

        logger.info("All Ray tasks have finished.")

        # 8) Save merged stats
        with self.logging_dir.open("stats.json", "wt") as statsfile:
            total_stats.save_to_disk(statsfile)

        logger.success(total_stats.get_repr(f"All {len(incomplete_ranks)} tasks"))
        return total_stats