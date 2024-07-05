import contextlib
from abc import ABC, abstractmethod
from typing import List, Tuple

from loguru import logger

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.batching import batched
from datatrove.utils.typeshelper import StatHints


def get_filter_result(res):
    result, reason = res, None
    if isinstance(result, tuple):
        result, reason = res
    return result, reason


class BaseFilter(PipelineStep, ABC):
    """Base module for Filters. Filters remove documents.

    Args:
        exclusion_writer: optionally pass in a writer that will save the dropped documents
    """

    type = "🔻 - FILTER"

    def __init__(self, exclusion_writer: DiskWriter = None, batch_size: int = 1):
        super().__init__()
        self.exclusion_writer = exclusion_writer
        self.batch_size = batch_size
        if self.batch_size > 1 and type(self).filter_batch == BaseFilter.filter_batch:
            logger.warning(f"{batch_size=} > 1 but {self} does not implement a custom filter_batch method.")

    @abstractmethod
    def filter(self, doc: Document) -> bool | Tuple[bool, str]:
        """Filter modules main method, for a single document
        Returns true if a sample should be KEPT, false if it should be REMOVED.

        Args:
            doc: sample to filter

        Returns:
            bool - whether the doc should be kept
            or (False, str), to drop with a specific reason
        """
        raise NotImplementedError

    def filter_batch(self, batch: List[Document]) -> List[bool | Tuple[bool, str]]:
        """
        Overwrite this method to implement batched filtering. Batches have size `self.batch_size`, except possibly the last one.
        Args:
            batch: a list of Document to process

        Returns: a list, the same size as `batch`, containing the filter result for each document

        """
        return list(map(self.filter, batch))

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
            for batch in batched(data, self.batch_size):
                if self.batch_size > 1:
                    self.stat_update("batches")
                with self.track_time("batch" if self.batch_size > 1 else None):
                    batch_filter_result = self.filter_batch(batch)
                for doc, doc_filter_result in zip(batch, batch_filter_result):
                    self.stat_update(StatHints.total)
                    filter_result, reason = get_filter_result(doc_filter_result)
                    if filter_result:
                        self.stat_update(StatHints.forwarded)
                        self.update_doc_stats(doc)
                        yield doc
                    else:
                        self.stat_update(StatHints.dropped)
                        if reason:
                            self.stat_update(f"dropped_{reason}")
                        if self.exclusion_writer:
                            if reason:
                                doc.metadata["filter_reason"] = reason
                            writer.write(doc, rank)
