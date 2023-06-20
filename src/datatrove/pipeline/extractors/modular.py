import re
from datatrove.data import Document
from .base import BaseExtractor

from readability import Document as ReadDocument
from inscriptis import get_text

from inscriptis.css_profiles import CSS_PROFILES
from inscriptis.model.config import ParserConfig

INSCRIPTIS_CONFIG = ParserConfig(css=CSS_PROFILES["strict"])


class ReadabilityInscriptis(BaseExtractor):
    """

    """

    def __init__(
            self,

            timeout: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.timeout = timeout
        self.regex_excessive_lines = re.compile(r"\n\s*\n")

    def __repr__(self):
        " ".join([super().__repr__(), "readability + inscriptis"])

    def extract(self, doc: Document) -> bool:
        content = doc.content
        parsed_doc = ReadDocument(content)
        clean_html = parsed_doc.summary(html_partial=True)
        content = get_text(clean_html, INSCRIPTIS_CONFIG).strip()
        # remove excessive empty lines
        content = self.regex_excessive_lines.sub("\n\n", content)
        if content:
            doc.content = content
            return True
        return False
