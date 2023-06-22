import os

from datatrove.executor.base import PipelineExecutor


class SlurmPipelineExecutor(PipelineExecutor):
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
        if "SLURM_JOB_ID" in os.environ:
            rank = 0 # TODO get it from env
            self._run_for_rank(rank)
        else:
            pass
            # TODO launch the job

    @property
    def world_size(self):
        return self.tasks
