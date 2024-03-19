import re

from .base import BaseExtractor


class ReadabilityInscriptis(BaseExtractor):
    """
    Extracts the text from the HTML document using readability and inscriptis.

    Args:
        max_new_lines: maximum number of consecutive \n to keep.
        min_text_length: the minimum string length of a text block. If all text blocks are shorter than
    `min_text_length`, the document is considered empty.
        min_text_score: `score = sqrt(block_lenth - min_text_length)`. The sum of scores of all text blocks must
    be greater than `min_text_score`.
        timeout: the timeout for extraction, per document, in seconds
    """

    _requires_dependencies = [
        "inscriptis",
        ("readability", "readability-lxml @ git+https://github.com/huggingface/python-readability.git@speedup"),
    ]

    def __init__(self, max_new_lines: int = 2, min_text_length=25, min_text_score=20, timeout: float = 0.1):
        from inscriptis.css_profiles import CSS_PROFILES
        from inscriptis.model.config import ParserConfig

        super().__init__(timeout)
        self.min_text_length = min_text_length
        self.min_text_score = min_text_score
        self.new_line_chars = "\n" * max_new_lines
        self.regex_excessive_lines = re.compile(r"(" + self.new_line_chars + "\n+)")
        self._parser_config = ParserConfig(css=CSS_PROFILES["strict"])

    def extract(self, text: str) -> str:
        """Extracts the text from the HTML document using readability and inscriptis.

        Args:
            the HTML document

        Returns:
            the extracted text
        """
        from inscriptis import get_text
        from readability import Document as _Document

        parsed_doc = _Document(text, min_text_length=self.min_text_length, min_text_score=self.min_text_score)
        clean_html = parsed_doc.summary(html_partial=True)
        text = get_text(clean_html, self._parser_config).strip()
        # remove excessive empty lines
        return self.regex_excessive_lines.sub(self.new_line_chars, text)
