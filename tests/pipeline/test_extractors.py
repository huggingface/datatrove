import unittest

from datatrove.data import Document
from datatrove.pipeline.extractors import ReadabilityInscriptis, Trafilatura


class TextExtraction(unittest.TestCase):
    ARTICLE_HTML = "<html><body><article><p>Hello World!</p></article></body></html>"

    def test_basic_article_trafilatura(self):
        extractor = Trafilatura()
        doc = Document(content=self.ARTICLE_HTML, data_id="0")
        extractor.extract(doc)
        self.assertEqual(doc.content, "Hello World!")

    def test_basic_article_readability(self):
        extractor = ReadabilityInscriptis()
        doc = Document(content=self.ARTICLE_HTML, data_id="0")
        extractor.extract(doc)
        self.assertEqual(doc.content, "Hello World!")
