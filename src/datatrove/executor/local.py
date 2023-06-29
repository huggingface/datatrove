from copy import deepcopy

import multiprocess.pool
from multiprocess import Semaphore

from datatrove.executor.base import PipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.stats import merge_all

download_semaphore, upload_semaphore = None, None


def init_pool_processes(dl_sem, up_sem):
    global download_semaphore, upload_semaphore
    download_semaphore = dl_sem
    upload_semaphore = up_sem


class LocalPipelineExecutor(PipelineExecutor):
    def __init__(
            self,
            tasks: int,
            workers: int = -1,
            max_concurrent_uploads: int = 20,
            max_concurrent_downloads: int = 50,
            save_stats: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.workers = workers if workers != -1 else tasks
        self.max_concurrent_uploads = max_concurrent_uploads
        self.max_concurrent_downloads = max_concurrent_downloads
        self.save_stats = save_stats

    def _run_for_rank(self, rank: int):
        if self.workers > 1:
            for pipeline_step in self.pipeline:
                if isinstance(pipeline_step, PipelineStep):
                    pipeline_step.set_up_dl_locks(download_semaphore, upload_semaphore)
        return super()._run_for_rank(rank)

    def run(self):
        if self.workers == 1:
            pipeline = self.pipeline
            stats = []
            for rank in range(self.tasks):
                self.pipeline = deepcopy(pipeline)
                stats.append(self._run_for_rank(rank))
        else:
            dl_sem = Semaphore(self.max_concurrent_downloads)
            up_sem = Semaphore(self.max_concurrent_uploads)
            with multiprocess.Pool(self.workers, initializer=init_pool_processes,
                                   initargs=(dl_sem, up_sem)) as pool:
                stats = list(pool.map(self._run_for_rank, range(self.tasks)))

        return merge_all(stats, save=self.save_stats)

    @property
    def world_size(self):
        return self.tasks
