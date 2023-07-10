from abc import ABC, abstractmethod

from datatrove.data import DocumentsPipeline
from datatrove.utils.stats import Stats


class PipelineStep(ABC):
    name: str = None
    type: str = None

    def __init__(self, **kwargs):
        self.stats = Stats(f"{self.__repr__()}")

    def stat_update(self, key, value: int = 1):
        self.stats.counter[key] += value

    def set_up_dl_locks(self, dl_lock, up_lock):
        pass

    def __repr__(self):
        return f"{self.type} --> {self.name}"

    @abstractmethod
    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data
