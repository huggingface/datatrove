import contextlib
from abc import ABC, abstractmethod
from typing import Tuple

from datatrove.data import DocumentsPipeline, Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter

from datatrove.utils.typeshelper import StatHints


def get_filter_result(res):
    result, reason = res, None
    if isinstance(result, tuple):
        result, reason = res
    return result, reason


class BaseFilter(PipelineStep, ABC):

    def __init__(
            self,
            exclusion_writer: DiskWriter = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.exclusion_writer = exclusion_writer
        self.type = "🔻 - FILTER"

    @abstractmethod
    def filter(self, doc: Document) -> bool | Tuple[bool, str]:
        """
        Filter modules' main method.
        Returns true if a sample should be filtered.

        @param doc: sample to (maybe) filter
        @return: bool - whether the doc should be filtered
        """
        raise NotImplementedError

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        step method for Filters.
        Drops documents that if .filter() is False

        @param datapipe: input DocumentsPipeline
        @return: DocumentsPipeline
        """

        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
            for doc in data:
                self.stat_update(StatHints.total)
                filter_result, reason = get_filter_result(self.filter(doc))
                if filter_result is True:
                    self.stat_update(StatHints.forwarded)
                    yield doc
                else:
                    self.stat_update(StatHints.dropped)
                    if self.exclusion_writer:
                        if reason:
                            doc.metadata["filter_reason"] = reason
                        writer.write(doc, rank)
