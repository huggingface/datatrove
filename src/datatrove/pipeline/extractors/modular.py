import re

from inscriptis import get_text
from inscriptis.css_profiles import CSS_PROFILES
from inscriptis.model.config import ParserConfig
from readability import Document as ReadDocument

from datatrove.data import Document

from .base import BaseExtractor


INSCRIPTIS_CONFIG = ParserConfig(css=CSS_PROFILES["strict"])


class ReadabilityInscriptis(BaseExtractor):
    """
    Extracts the text from the HTML document using readability and inscriptis.
    """

    def __init__(self, timeout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.timeout = timeout
        self.regex_excessive_lines = re.compile(r"\n\s*\n")

    def __repr__(self):
        " ".join([super().__repr__(), "readability + inscriptis"])

    def extract(self, doc: Document, min_text_length=25, min_text_score=20) -> bool:
        """
        :param doc: the document to extract the text from
        :param min_text_length: the minimum string length of a text block. If all text blocks are shorter than
        `min_text_length`, the document is considered empty.
        :param min_text_score: `score = sqrt(block_lenth - min_text_length)`. The sum of scores of all text blocks must
        be greater than `min_text_score`.
        """
        parsed_doc = ReadDocument(doc.content, min_text_length=min_text_length, min_text_score=min_text_score)
        clean_html = parsed_doc.summary(html_partial=True)
        content = get_text(clean_html, INSCRIPTIS_CONFIG).strip()
        # remove excessive empty lines
        content = self.regex_excessive_lines.sub("\n\n", content)
        if content:
            doc.content = content
            return True
        return False
