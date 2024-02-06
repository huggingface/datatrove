import re

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class RegexFilter(BaseFilter):
    name = "🕵 Regex"

    def __init__(self, regex_exp: str, exclusion_writer: DiskWriter = None):
        """
        filters if regex finds at least one match

        Args:
            regex_exp: regex expression
            exclusion_writer:
        """
        super().__init__(exclusion_writer)
        self.regex = re.compile(regex_exp)

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        return not self.regex.search(doc.text)
