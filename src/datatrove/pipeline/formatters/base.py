from abc import ABC, abstractmethod

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


class BaseFormatter(PipelineStep, ABC):
    type = "✂️ - FORMAT"

    def __init__(self):
        super().__init__()

    @abstractmethod
    def format(self, text: str) -> str:
        return text

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            self.stat_update(StatHints.total)
            with self.track_time():
                doc.text = self.format(doc.text)
            yield doc
