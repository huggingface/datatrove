from nltk.tokenize import word_tokenize

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class ListFilter(BaseFilter):
    name = "🎅 List"

    def __init__(self, new_line_ratio: float | None = 0.3, exclusion_writer: DiskWriter = None):  # TODO better tune
        """ """
        super().__init__(exclusion_writer)
        self.new_line_ratio = new_line_ratio

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Applies heuristic rules to decide if a document should be REMOVED
        Args:
            doc

        Returns:
            False if sample.text is a list
        """
        text = doc.text
        words = word_tokenize(text)  # TODO we should use language id filter
        new_line = text.count("\n")
        if new_line / len(words) > self.new_line_ratio:
            return False, "Suspected list"

        return True
