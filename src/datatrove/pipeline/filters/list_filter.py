from nltk.tokenize import word_tokenize

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter


class ListFilter(BaseFilter):
    name = "🎅 List"

    def __init__(
        self,
        new_line_ratio: float | None = 0.3,  # TODO better tune
        **kwargs,
    ):
        """ """
        super().__init__(**kwargs)
        self.new_line_ratio = new_line_ratio

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """
            Applies the heuristics rules to decide if a document should be REMOVED:
                -

        :param doc
        :return: False if sample.content is a list
        """
        text = doc.content
        words = word_tokenize(text)  # TODO we should use language id filter
        new_line = text.count("\n")
        if new_line / len(words) > self.new_line_ratio:
            return False, "Suspected list"

        return True
