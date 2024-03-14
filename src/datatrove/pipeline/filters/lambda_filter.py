from typing import Callable

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class LambdaFilter(BaseFilter):
    name = "👤 Lambda"

    def __init__(self, filter_function: Callable[[Document], bool], exclusion_writer: DiskWriter = None):
        """
        filters documents triggering the given filter_function with respect to a specific metadata key.

        Args:
            filter_function:
            exclusion_writer:
        """
        super().__init__(exclusion_writer)
        self.filter_function = filter_function

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        return self.filter_function(doc)
