from copy import deepcopy
from typing import Callable

import multiprocess.pool
from multiprocess import Queue, Semaphore

from datatrove.executor.base import PipelineExecutor
from datatrove.io import BaseOutputDataFolder
from datatrove.pipeline.base import PipelineStep


download_semaphore, upload_semaphore = None, None


def init_pool_processes(dl_sem, up_sem):
    global download_semaphore, upload_semaphore
    download_semaphore = dl_sem
    upload_semaphore = up_sem


class LocalPipelineExecutor(PipelineExecutor):
    def __init__(
        self,
        pipeline: list[PipelineStep | Callable],
        tasks: int = 1,
        workers: int = -1,
        max_concurrent_uploads: int = 20,
        max_concurrent_downloads: int = 50,
        logging_dir: BaseOutputDataFolder = None,
    ):
        super().__init__(pipeline, logging_dir)
        self.local_ranks = None
        self.tasks = tasks
        self.workers = workers if workers != -1 else tasks
        self.max_concurrent_uploads = max_concurrent_uploads
        self.max_concurrent_downloads = max_concurrent_downloads

    def _run_for_rank(self, rank: int, local_rank: int = -1):
        if self.workers > 1:
            for pipeline_step in self.pipeline:
                if isinstance(pipeline_step, PipelineStep):
                    pipeline_step.set_up_dl_locks(download_semaphore, upload_semaphore)
        local_rank = self.local_ranks.get()
        try:
            return super()._run_for_rank(rank, local_rank)
        finally:
            self.local_ranks.put(local_rank)  # free up used rank

    def run(self):
        self.local_ranks = Queue()
        for i in range(self.workers):
            self.local_ranks.put(i)

        if self.workers == 1:
            pipeline = self.pipeline
            stats = []
            for rank in range(self.tasks):
                self.pipeline = deepcopy(pipeline)
                stats.append(self._run_for_rank(rank))
        else:
            dl_sem = Semaphore(self.max_concurrent_downloads)
            up_sem = Semaphore(self.max_concurrent_uploads)

            with multiprocess.Pool(self.workers, initializer=init_pool_processes, initargs=(dl_sem, up_sem)) as pool:
                stats = list(pool.map(self._run_for_rank, range(self.tasks)))
        return sum(stats)

    @property
    def world_size(self):
        return self.tasks
