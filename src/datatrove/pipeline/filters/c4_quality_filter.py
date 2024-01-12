import heapq
import re

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class C4QualityFilter(BaseFilter):
    """Applies heuristic rules from C4 https://jmlr.org/papers/volume21/20-074/20-074.pdf

    - We only retained lines that ended in a terminal punctuation mark (! . " ?)
    - We discarded any page with fewer than 5 sentences and only retained lines that contained at least 3 words
    - We removed any page that contained any word on the “List of Dirty, Naughty, Obscene or Otherwise Bad Words”
    - We removed any line with the word Javascript.
    - We removed any page where the phrase “lorem ipsum” appeared
    - We removed any pages that contained a curly bracket
    """

    name = "⛰ C4 Quality"
    _requires_dependencies = ["nltk"]

    def __init__(self, exclusion_writer: DiskWriter = None):
        from nltk import load

        super().__init__(exclusion_writer)

        self.lorem_ipsum = re.compile(r"(?i)lorem ipsum")
        self.javascript = re.compile(r"(?i)javascript")
        self.min_lines = 5
        self.min_words = 3
        self.stop_chars = (".", "'", '"', "!", "?")
        self._tokenizer = load("tokenizers/punkt/english.pickle")

    def line_filter(self, line: str):
        if self.javascript.search(line):
            return False
        if not line.endswith(self.stop_chars):
            return False
        if len(line.split()) < self.min_words:
            return False

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Args:
            doc

        Returns:

        """
        from nltk.tokenize import sent_tokenize

        lines = sent_tokenize(doc.content)
        if len(lines) < self.min_lines:
            return False, f"< {self.min_lines} lines"

        if "{" or "}" in doc.content:
            return False, "curly brackets"

        if self.lorem_ipsum.search(doc.content):
            return False, "lorem ipsum"

        # TODO find a way to track skip lines to re-insert them when joining lines.
        # sent_dedup's remove_dup_sentences may be adapted for this
        lines = [line for line in lines if self.line_filter(line)]
        doc.content = " ".join(lines)


class C4ParagraphFilter(BaseFilter):
    """Applies paragraph filtering from mC4

    https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/c4_utils.py#L551
    """

    name = "⛰ C4 Paragraph"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)

        self.min_paragraphs = 3
        self.min_paragraph_len = 200
        self.line_delimiter = "\n"

    def paragraph_filter(self, page):
        """Returns False iff a page has too few or too short paragraphs."""
        lines = page.split(self.line_delimiter)
        # Filter out docs that don't have at least three "paragraphs"
        # (lines >= `min_paragraph_len` chars).
        if (
            len(lines) < self.min_paragraphs
            or min(heapq.nlargest(3, [len(line) for line in lines])) < self.min_paragraph_len
        ):
            return False
        return True

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Args:
            doc

        Returns:

        """
        if not self.paragraph_filter(doc.content):
            return False, f"< {self.min_paragraphs} paragraphs"
        return True
