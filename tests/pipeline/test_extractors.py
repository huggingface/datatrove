import unittest

from datatrove.pipeline.extractors import ReadabilityInscriptis, Trafilatura


ARTICLE_HTML = "<html><body><article><p>Hello World!</p></article></body></html>"


class TestExtractors(unittest.TestCase):
    def test_basic_article_trafilatura(self):
        extractor = Trafilatura()
        self.assertEqual(extractor.extract(ARTICLE_HTML), "Hello World!")

    def test_basic_article_readability(self):
        extractor = ReadabilityInscriptis(min_text_length=10, min_text_score=1)
        self.assertEqual(extractor.extract(ARTICLE_HTML), "Hello World!")
