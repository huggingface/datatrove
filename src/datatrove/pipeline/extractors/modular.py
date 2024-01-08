import re

from inscriptis import get_text
from inscriptis.css_profiles import CSS_PROFILES
from inscriptis.model.config import ParserConfig
from readability import Document as ReadDocument

from .base import BaseExtractor


INSCRIPTIS_CONFIG = ParserConfig(css=CSS_PROFILES["strict"])


class ReadabilityInscriptis(BaseExtractor):
    """
    Extracts the text from the HTML document using readability and inscriptis.
    """

    def __init__(self, max_new_lines: int = 2, min_text_length=25, min_text_score=20, timeout: float = 0.1):
        """
        :param max_new_lines: maximum number of consecutive \n to keep.
        :param min_text_length: the minimum string length of a text block. If all text blocks are shorter than
        `min_text_length`, the document is considered empty.
        :param min_text_score: `score = sqrt(block_lenth - min_text_length)`. The sum of scores of all text blocks must
        be greater than `min_text_score`.
        :param timeout: the timeout for extraction, per document, in seconds
        """
        super().__init__(timeout)
        self.min_text_length = min_text_length
        self.min_text_score = min_text_score
        self.new_line_chars = "\n" * max_new_lines
        self.regex_excessive_lines = re.compile(r"(" + self.new_line_chars + "\n+)")

    def extract(self, text: str) -> str:
        parsed_doc = ReadDocument(text, min_text_length=self.min_text_length, min_text_score=self.min_text_score)
        clean_html = parsed_doc.summary(html_partial=True)
        text = get_text(clean_html, INSCRIPTIS_CONFIG).strip()
        # remove excessive empty lines
        return self.regex_excessive_lines.sub(self.new_line_chars, text)
