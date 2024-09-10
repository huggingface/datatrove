from abc import ABC, abstractmethod
from typing import List

from loguru import logger

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.batching import batched
from datatrove.utils.typeshelper import StatHints


class BaseEnrisher(PipelineStep, ABC):
    """Base module for Enrishers. Enrishers enrich documents with additional fields.

    Args:
        batch_size: size of the batches to process
    """

    type = "➕ - ENRISHER"

    def __init__(self, batch_size: int = 1):
        super().__init__()
        self.batch_size = batch_size
        if self.batch_size > 1 and type(self).enrish_batch == BaseEnrisher.enrish_batch:
            logger.warning(f"{batch_size=} > 1 but {self} does not implement a custom enrish_batch method.")

    @abstractmethod
    def enrish(self, doc: Document) -> Document:
        """Enrishers main method, for a single document

        Args:
            doc: sample to enrich

        Returns:
            Document - the enriched document
        """
        raise NotImplementedError

    def enrish_batch(self, batch: List[Document]) -> List[Document]:
        """
        Overwrite this method to implement batched enrichment. Batches have size `self.batch_size`, except possibly the last one.
        Args:
            batch: a list of Document to process

        Returns: a list, the same size as `batch`, containing the enriched documents
        """
        return list(map(self.enrish, batch))

    def run(self, data: DocumentsPipeline, rank, world_size) -> DocumentsPipeline:
        """
        Run the Enrisher on the data

        Args:
            data: the data to process

        Returns:
            the processed data
        """
        for batch in batched(data, self.batch_size):
            if self.batch_size > 1:
                self.stat_update("batches")
            with self.track_time("batch" if self.batch_size > 1 else None):
                batch_enrish_result = self.enrish_batch(batch)
            for enrished_doc in batch_enrish_result:
                self.stat_update(StatHints.total)
                self.update_doc_stats(enrished_doc)
                yield enrished_doc
