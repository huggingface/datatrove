from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List

from loguru import logger

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.batching import batched
from datatrove.utils.typeshelper import StatHints


class BaseEnricher(PipelineStep, ABC):
    """Base module for Enrichers. Enrichers enrich documents with additional fields.

    Args:
        batch_size: size of the batches to process
    """

    type = "➕ - ENRICHER"

    def __init__(self, batch_size: int = 1):
        super().__init__()
        self.batch_size = batch_size
        if self.batch_size > 1 and type(self).enrich_batch == BaseEnricher.enrich_batch:
            logger.warning(f"{batch_size=} > 1 but {self} does not implement a custom enrich_batch method.")

    @abstractmethod
    def enrich(self, doc: Document) -> Document:
        """Enrichers main method, for a single document

        Args:
            doc: sample to enrich

        Returns:
            Document - the enriched document
        """
        raise NotImplementedError

    def enrich_batch(self, batch: List[Document]) -> List[Document]:
        """
        Overwrite this method to implement batched enrichment. Batches have size `self.batch_size`, except possibly the last one.
        Args:
            batch: a list of Document to process

        Returns: a list, the same size as `batch`, containing the enriched documents
        """
        with ThreadPoolExecutor(max_workers=8) as executor:
            enriched_docs = list(executor.map(self.enrich, batch))
        return enriched_docs

    def run(self, data: DocumentsPipeline, rank, world_size) -> DocumentsPipeline:
        """
        Run the Enricher on the data

        Args:
            data: the data to process

        Returns:
            the processed data
        """
        for batch in batched(data, self.batch_size):
            if self.batch_size > 1:
                self.stat_update("batches")
            with self.track_time("batch" if self.batch_size > 1 else None):
                batch_enrich_result = self.enrich_batch(batch)
            for enriched_doc in batch_enrich_result:
                self.stat_update(StatHints.total)
                self.update_doc_stats(enriched_doc)
                yield enriched_doc
