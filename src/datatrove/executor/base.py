from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Callable

from datatrove.pipeline.base import PipelineStep


class PipelineExecutor(ABC):
    @abstractmethod
    def __init__(
            self,
            pipeline: list[PipelineStep | Callable]
    ):
        self.pipeline: list[PipelineStep | Callable] = pipeline

    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def world_size(self):
        return 0

    def _run_for_rank(self, rank: int):
        pipeline = deepcopy(self.pipeline)
        # pipe data from one step to the next
        pipelined_data = None
        for pipeline_step in pipeline:
            pipelined_data = pipeline_step(pipelined_data, rank, self.world_size)
        if pipelined_data:
            deque(pipelined_data, maxlen=0)
