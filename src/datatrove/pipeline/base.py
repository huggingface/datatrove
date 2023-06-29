from abc import ABC, abstractmethod

from datatrove.data import DocumentsPipeline
from datatrove.utils.stats import TimeStatsManager, Stats

from collections import Counter


class PipelineStep(ABC):
    name: str = None
    type: str = None

    def __init__(self, **kwargs):
        self._stats = Counter()
        self.time_stats_manager = TimeStatsManager()

    def stat_update(self, key, value: int = 1):
        self._stats[key] += value

    def set_up_dl_locks(self, dl_lock, up_lock):
        pass

    def __repr__(self):
        return " --> ".join([self.type, self.name])

    def stats(self) -> Stats:
        return Stats(f"{self.__repr__()}", self.time_stats_manager.get_stats(), self._stats)

    @abstractmethod
    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data
