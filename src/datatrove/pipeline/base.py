from abc import ABC, abstractmethod

from datatrove.data import Document, DocumentsPipeline
from datatrove.utils.stats import Stats


class PipelineStep(ABC):
    name: str = None
    type: str = None

    def __init__(self):
        self.stats = Stats(str(self))

    def stat_update(self, *labels, value: int = 1, unit: str = None):
        for label in labels:
            self.stats[label].update(value, unit)

    def update_doc_stats(self, document: Document):
        self.stats["doc_len"] += len(document.text)
        if token_count := document.metadata.get("token_count", None):
            self.stats["doc_len_tokens"] += token_count

    def track_time(self, unit: str = None):
        if unit:
            self.stats.time_stats.unit = unit
        return self.stats.time_stats

    def __repr__(self):
        return f"{self.type}: {self.name}"

    @abstractmethod
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data

    def __call__(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        return self.run(data, rank, world_size)
