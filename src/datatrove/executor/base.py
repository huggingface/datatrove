from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from typing import Callable

from datatrove.pipeline.base import PipelineStep
from collections import Counter


class PipelineExecutor(ABC):
    @abstractmethod
    def __init__(
            self,
            pipeline: list[PipelineStep | Callable],
            save_stats: bool = False
    ):
        self.pipeline: list[PipelineStep | Callable] = pipeline
        self.save_stats = save_stats

        pipeline = "\n".join([pipe.__repr__() for pipe in self.pipeline])
        print(f"--- 🛠️PIPELINE 🛠\n{pipeline}")

    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def world_size(self):
        return 0

    def _run_for_rank(self, rank: int):
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
        stats = [pipeline_step.stats() for pipeline_step in self.pipeline]
        return stats
