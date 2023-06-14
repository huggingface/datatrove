from abc import ABC, abstractmethod

from datatrove.data import DocumentsPipeline


class PipelineStep(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, rank: int, world_size: int, data: DocumentsPipeline) -> DocumentsPipeline:
        if data:
            yield from data
