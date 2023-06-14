from collections import deque
from concurrent.futures import ProcessPoolExecutor

from datatrove.executor.base import PipelineExecutor


class LocalPipelineExecutor(PipelineExecutor):
    def __init__(
            self,
            tasks: int,
            workers: int = -1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.workers = workers if workers != -1 else tasks

    def run(self):
        if self.workers == 1:
            for rank in range(self.tasks):
                self._run_for_rank(rank)
        else:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                deque(pool.map(self._run_for_rank, range(self.tasks)), maxlen=0)

    @property
    def world_size(self):
        return self.tasks
