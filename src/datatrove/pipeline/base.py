from abc import ABC, abstractmethod

from datatrove.data import DocumentsPipeline


class PipelineStep(ABC):

    def set_up_dl_locks(self, dl_lock, up_lock):
        pass

    @abstractmethod
    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data
