import contextlib
from abc import ABC, abstractmethod
from typing import Tuple

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.typeshelper import StatHints


def get_filter_result(res):
    result, reason = res, None
    if isinstance(result, tuple):
        result, reason = res
    return result, reason


class BaseFilter(PipelineStep, ABC):
    """Base module for Filters. Filters remove documents."""

    type = "🔻 - FILTER"

    def __init__(self, exclusion_writer: DiskWriter = None):
        """
        :param exclusion_writer: optionally pass in a writer that will save the dropped documents
        """
        super().__init__()
        self.exclusion_writer = exclusion_writer

    @abstractmethod
    def filter(self, doc: Document) -> bool | Tuple[bool, str]:
        """Filter modules main method.
        Returns true if a sample should be kept, false if it should be removed.

        Args:
            doc: sample to filter

        Returns:
            bool - whether the doc should be kept
            or (False, str), to drop with a specific reason
        """
        raise NotImplementedError

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
            for doc in data:
                self.stat_update(StatHints.total)
                with self.track_time():
                    filter_result, reason = get_filter_result(self.filter(doc))
                    if filter_result:
                        self.stat_update(StatHints.forwarded)
                        self.stats.doc_len_stats += len(doc.content)
                    else:
                        self.stat_update(StatHints.dropped)
                        if self.exclusion_writer:
                            if reason:
                                doc.metadata["filter_reason"] = reason
                            writer.write(doc, rank)
                        continue
                yield doc
