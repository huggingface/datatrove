from typing import Callable

from datatrove.data import Document
from datatrove.pipeline.enrichers.base_enricher import BaseEnricher


class LambdaEnricher(BaseEnricher):
    name = "👤 Lambda Enricher"

    def __init__(
        self,
        enricher_function: Callable[[Document], Document],
    ):
        """
        Enrich/Modify documents triggering the given enricher_function with respect to a specific metadata key.

        Args:
            enricher_function:
            exclusion_writer:
        """
        super().__init__()
        self.enricher_function = enricher_function

    def enrich(self, doc: Document) -> Document:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        return self.enricher_function(doc)
