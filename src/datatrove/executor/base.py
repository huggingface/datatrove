from abc import ABC, abstractmethod
from collections import deque

from datatrove.pipeline.base import PipelineStep


class PipelineExecutor(ABC):
    @abstractmethod
    def __init__(
            self,
            pipeline: list[PipelineStep]
    ):
        self.pipeline: list[PipelineStep] = pipeline

    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def world_size(self):
        return 0

    def _run_for_rank(self, rank: int):
        pipelined_data = None
        for pipeline_step in self.pipeline:
            pipelined_data = pipeline_step(pipelined_data, rank, self.world_size)
        if pipelined_data:
            deque(pipelined_data, maxlen=0)
