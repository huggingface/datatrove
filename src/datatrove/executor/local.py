from copy import deepcopy
from multiprocessing import Value
from typing import Callable, List, Optional, Union

import multiprocess.pool
from loguru import logger
from multiprocess import Queue, Semaphore

from datatrove.executor.base import PipelineExecutor
from datatrove.io import BaseOutputDataFolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.stats import PipelineStats


# multiprocessing vars
download_semaphore, upload_semaphore, ranks_queue, completed = None, None, None, None


def init_pool_processes(dl_sem, up_sem, ranks_q, completed_counter):
    global download_semaphore, upload_semaphore, ranks_queue, completed
    download_semaphore = dl_sem
    upload_semaphore = up_sem
    ranks_queue = ranks_q
    completed = completed_counter


class LocalPipelineExecutor(PipelineExecutor):
    def __init__(
        self,
        pipeline: List[Union[PipelineStep, Callable]],
        tasks: int = 1,
        workers: int = -1,
        max_concurrent_uploads: int = 20,
        max_concurrent_downloads: int = 50,
        logging_dir: Optional[Union[BaseOutputDataFolder, str]] = None,
        skip_completed: bool = True,
    ):
        """Execute a pipeline locally

        Args:
            pipeline: a list of PipelineStep and/or custom functions
                with arguments (data: DocumentsPipeline, rank: int,
                world_size: int)
            tasks: total number of tasks to run the pipeline on
            workers: how many tasks to run simultaneously. -1 for no
                limit
            max_concurrent_uploads: limit the number of files that may
                be uploaded simultaneously to avoid rate limits
            max_concurrent_downloads: limit the number of files that may
                be downloaded simultaneously to avoid rate limits
            logging_dir: where to save logs, stats, etc. Should be an
                OutputDataFolder or a str. If str, BaseOutputDataFolder.from_path(value) will be used to convert it
            skip_completed: whether to skip tasks that were completed in
                previous runs. default: True
        """
        super().__init__(pipeline, logging_dir, skip_completed)
        self.tasks = tasks
        self.workers = workers if workers != -1 else tasks
        self.max_concurrent_uploads = max_concurrent_uploads
        self.max_concurrent_downloads = max_concurrent_downloads

    def _run_for_rank(self, rank: int, local_rank: int = -1):
        if self.workers > 1:
            for pipeline_step in self.pipeline:
                if isinstance(pipeline_step, PipelineStep):
                    pipeline_step.set_up_dl_locks(download_semaphore, upload_semaphore)
        local_rank = ranks_queue.get()
        try:
            return super()._run_for_rank(rank, local_rank)
        finally:
            if completed:
                with completed.get_lock():
                    completed.value += 1
                    logger.info(f"{completed.value}/{self.world_size} tasks completed.")
            ranks_queue.put(local_rank)  # free up used rank

    def run(self):
        if all(map(self.is_rank_completed, range(self.tasks))):
            logger.info(f"Not doing anything as all {self.tasks} tasks have already been completed.")
            return

        self.save_executor_as_json()
        ranks_q = Queue()
        for i in range(self.workers):
            ranks_q.put(i)

        ranks_to_run = self.get_incomplete_ranks()
        if (skipped := self.tasks - len(ranks_to_run)) > 0:
            logger.info(f"Skipping {skipped} already completed tasks")

        if self.workers == 1:
            global ranks_queue
            ranks_queue = ranks_q
            pipeline = self.pipeline
            stats = []
            for rank in ranks_to_run:
                self.pipeline = deepcopy(pipeline)
                stats.append(self._run_for_rank(rank))
        else:
            dl_sem = Semaphore(self.max_concurrent_downloads)
            up_sem = Semaphore(self.max_concurrent_uploads)
            completed_counter = Value("i", skipped)

            with multiprocess.Pool(
                self.workers, initializer=init_pool_processes, initargs=(dl_sem, up_sem, ranks_q, completed_counter)
            ) as pool:
                stats = list(pool.imap_unordered(self._run_for_rank, ranks_to_run))
        # merged stats
        stats = sum(stats, start=PipelineStats())
        stats.save_to_disk(self.logging_dir.open("stats.json"))
        logger.success(stats.get_repr(f"All {self.tasks} tasks"))
        self.logging_dir.close()
        return stats

    @property
    def world_size(self):
        return self.tasks
