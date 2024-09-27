import unittest

from datatrove.pipeline.extractors import (
    Inscriptis,
    Justext,
    ReadabiliPy,
    Readability,
    ReadabilityInscriptis,
    Resiliparse,
    Trafilatura,
)

from ..utils import (
    require_inscriptis,
    require_justext,
    require_readabilipy,
    require_readability,
    require_resiliparse,
    require_trafilatura,
)


ARTICLE_HTML = "<html><body><article><p>Hello World!</p></article></body></html>"


class TestExtractors(unittest.TestCase):
    @require_trafilatura
    def test_basic_article_trafilatura(self):
        extractor = Trafilatura()
        self.assertEqual(extractor.extract(ARTICLE_HTML), "Hello World!")

    @require_readability
    @require_inscriptis
    def test_basic_article_readability_inscriptis(self):
        extractor = ReadabilityInscriptis(min_text_length=10, min_text_score=1)
        self.assertEqual(extractor.extract(ARTICLE_HTML), "Hello World!")

    @require_justext
    def test_basic_article_justext(self):
        extractor = Justext(remove_boilerplate=False)
        self.assertEqual(extractor.extract(ARTICLE_HTML), "Hello World!")

    @require_resiliparse
    def test_basic_article_resiliparse(self):
        extractor = Resiliparse()
        self.assertEqual(extractor.extract(ARTICLE_HTML), "Hello World!")

    @require_readabilipy
    def test_basic_article_readabilipy(self):
        extractor = ReadabiliPy()
        self.assertEqual(extractor.extract(ARTICLE_HTML), "Hello World!")

    @require_inscriptis
    def test_basic_article_inscriptis(self):
        extractor = Inscriptis()
        self.assertEqual(extractor.extract(ARTICLE_HTML), "Hello World!")

    @require_readability
    def test_basic_article_readability(self):
        extractor = Readability(min_text_length=10, min_text_score=1)
        postprocessor = Trafilatura()
        self.assertEqual(extractor.extract(ARTICLE_HTML, postprocessor=postprocessor), "Hello World!")
