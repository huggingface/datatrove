from typing import Callable

from datatrove.data import Document
from datatrove.pipeline.enrishers.base_enrisher import BaseEnrisher


class LambdaEnrisher(BaseEnrisher):
    name = "👤 Lambda Enrisher"

    def __init__(
        self,
        enrisher_function: Callable[[Document], Document],
    ):
        """
        Enrish/Modify documents triggering the given enrisher_function with respect to a specific metadata key.

        Args:
            enrisher_function:
            exclusion_writer:
        """
        super().__init__()
        self.enrisher_function = enrisher_function

    def enrish(self, doc: Document) -> Document:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        return self.enrisher_function(doc)
