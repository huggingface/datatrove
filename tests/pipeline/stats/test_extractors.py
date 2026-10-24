import unittest

from datatrove.data import Document
from datatrove.pipeline.extractors import ReadabilityInscriptis, Trafilatura
from datatrove.pipeline.extractors.base import BaseExtractor

from ...utils import require_inscriptis, require_readability, require_trafilatura


ARTICLE_HTML = "<html><body><article><p>Hello World!</p></article></body></html>"


class EmptyFailsExtractor(BaseExtractor):
    """Extractor that raises on empty input, matching readability's warmup failure."""

    name = "empty-fails"

    def __init__(self, timeout: float = 2.0):
        super().__init__(timeout)

    def extract(self, text: str) -> str:
        if text == "":
            raise ValueError("Document is empty")
        return text


class TestExtractors(unittest.TestCase):
    @require_trafilatura
    def test_basic_article_trafilatura(self):
        extractor = Trafilatura()
        self.assertEqual(extractor.extract(ARTICLE_HTML), "Hello World!")

    @require_readability
    @require_inscriptis
    def test_basic_article_readability(self):
        extractor = ReadabilityInscriptis(min_text_length=10, min_text_score=1)
        self.assertEqual(extractor.extract(ARTICLE_HTML), "Hello World!")

    @require_readability
    @require_inscriptis
    def test_readability_empty_html(self):
        extractor = ReadabilityInscriptis(min_text_length=10, min_text_score=1)
        self.assertEqual(extractor.extract(""), "")

    @require_readability
    @require_inscriptis
    def test_readability_run_survives_empty_warmup(self):
        extractor = ReadabilityInscriptis(min_text_length=10, min_text_score=1, timeout=2)
        result = list(extractor.run(iter([Document(text=ARTICLE_HTML, id="1")])))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Hello World!")

    def test_sandbox_warmup_survives_empty_failure(self):
        extractor = EmptyFailsExtractor(timeout=5)
        result = list(extractor.run(iter([Document(text="<p>hello</p>", id="1")])))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "<p>hello</p>")
