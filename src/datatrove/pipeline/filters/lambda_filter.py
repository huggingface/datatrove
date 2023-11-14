from typing import Callable

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class LambdaFilter(BaseFilter):
    name = "👤 Lambda"

    def __init__(self, filter_function: Callable[[Document], bool], exclusion_writer: DiskWriter = None):
        """
        filters documents triggering the given filter_function with respect to a specific metadata key.

        @param regex_exp: regex expression
        """
        super().__init__(exclusion_writer)
        self.filter_function = filter_function

    def filter(self, doc: Document) -> bool:
        """

        :param doc: document
        :return: is_filter
        """
        return self.filter_function(doc)
