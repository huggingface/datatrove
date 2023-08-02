import re

from nltk.tokenize import sent_tokenize

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.utils import get_language, nltk_warning_msg


"""
Applies heuristic rules from C4 https://jmlr.org/papers/volume21/20-074/20-074.pdf

- We only retained lines that ended in a terminal punctuation mark (! . " ?)
- We discarded any page with fewer than 5 sentences and only retained lines that contained at least 3 words
- We removed any page that contained any word on the “List of Dirty, Naughty, Obscene or Otherwise Bad Words”
- We removed any line with the word Javascript.
- We removed any page where the phrase “lorem ipsum” appeared
- We removed any pages that contained a curly bracket
"""


class C4QualityFilter(BaseFilter):
    def __init__(self, **kwargs):
        """ """
        super().__init__(**kwargs)

        self.name = "⛰️ C4 Quality"
        self.lorem_ipsum = re.compile(r"(?i)lorem ipsum")
        self.javascript = re.compile(r"(?i)javascript")
        self.min_lines = 5
        self.min_words = 3
        self.stop_chars = (".", "'", '"', "!", "?")
        self.warning_msg = True

    def line_filter(self, line: str):
        if self.javascript.search(line):
            return False
        if not line.endswith(self.stop_chars):
            return False
        if len(line.split()) < self.min_words:
            return False

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """

        :param doc:
        :return:
        """

        self.warning_msg = nltk_warning_msg(doc) if self.warning_msg else False

        lines = sent_tokenize(doc.content, language=get_language(doc))
        if len(lines) < self.min_lines:
            return False, f"< {self.min_lines} lines"

        if "{" or "}" in doc.content:
            return False, "curly brackets"

        if self.lorem_ipsum.search(doc.content):
            return False, "lorem ipsum"

        # TODO find a way to track skip lines to re-insert them when joining lines.
        lines = [line for line in lines if self.line_filter(line)]
        doc.content = " ".join(lines)
