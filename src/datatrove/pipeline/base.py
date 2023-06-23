from abc import ABC, abstractmethod

from datatrove.data import DocumentsPipeline

from collections import Counter


class PipelineStep(ABC):

    def __init__(self, **kwargs):
        self._stats = Counter()

    def stat_update(self, key, value: int = 1):
        self._stats[key] += value

    def set_up_dl_locks(self, dl_lock, up_lock):
        pass

    def stats(self) -> tuple[str, Counter]:
        return f"{str(type(self))}", self._stats  # SUPER UGLY, to change

    @abstractmethod
    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data
