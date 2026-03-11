import unittest

from datatrove.data import Document
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
    LambdaFilter,
    LanguageFilter,
    RegexFilter,
    UnigramLogProbFilter,
    URLFilter,
)
from datatrove.pipeline.filters.c4_filters import C4ParagraphFilter, C4QualityFilter
from datatrove.pipeline.filters.fineweb_quality_filter import FineWebQualityFilter
from datatrove.pipeline.filters.sampler_filter import SamplerFilter

from ..utils import require_fasttext, require_nltk, require_tldextract


TEXT_LF_1 = (
    "I wish it need not have happened in my time,' said Frodo. 'So do I,' said Gandalf, 'and so do all who live to "
    "see such times. But that is not for them to decide. All we have to decide is what to do with the time that is "
    "given us.'"
)

TEXT_LF_2 = (
    "Un magicien n'est jamais en retard Frodon Sacquet. Pas plus qu'il est en avance. Il arrive précisément "
    "à l'heure prévue."
)

TEXT_LF_3 = "Um mago nunca chega tarde, Frodo Bolseiro. Nem cedo. Ele chega precisamente na hora que pretende."

TEXT_LF_4 = (
    "Molti tra i vivi meritano la morte. E parecchi che sono morti avrebbero meritato la vita. Sei forse tu in "
    "grado di dargliela? E allora non essere troppo generoso nel distribuire la morte nei tuoi giudizi: "
    "sappi che nemmeno i più saggi possono vedere tutte le conseguenze."
)


def get_doc(text: str, url: str = "https://example.com") -> Document:
    return Document(text=text, id="0", metadata={"url": url})


class TestFilters(unittest.TestCase):
    def check_filter(self, filter, doc, filter_reason):
        filter_result = filter.filter(doc)
        self.assertEqual(type(filter_result), tuple)
        self.assertEqual(filter_result[1], filter_reason)

    @require_nltk
    def test_gopher_repetition(self):
        gopher_repetition = GopherRepetitionFilter()

        self.check_filter(gopher_repetition, get_doc("I am your father.\n" * 4), "dup_line_frac")
        self.check_filter(gopher_repetition, get_doc("I am your father.\n\n" * 4), "dup_para_frac")
        text = "I am groot.\n\n" + "You are a wizard.\n\n" + "I am your father.\n\n" + f"{'x' * 30}.\n\n" * 2
        self.check_filter(gopher_repetition, get_doc(text), "dup_para_char_frac")
        doc = get_doc("I am groot.\n" + "You are a wizard.\n" + "I am your father.\n" + f"{'x' * 40}.\n" * 2)
        self.check_filter(gopher_repetition, doc, "dup_line_char_frac")
        self.check_filter(gopher_repetition, get_doc("I am Frank, I am Frank, I am Frank"), "top_2_gram")
        doc = get_doc("I am Frank, you are Jhon. I am Frank. I am Frank you are Jhon")
        self.check_filter(gopher_repetition, doc, "top_3_gram")
        doc = get_doc("I am Frank, you are Jhon. I am Frank. I am Frank you are Jhon")
        self.check_filter(gopher_repetition, doc, "top_3_gram")
        doc = get_doc("I am a solo traveller " * 4 + TEXT_LF_1)
        self.check_filter(gopher_repetition, doc, "duplicated_5_n_grams")

    def test_gopher_quality(self):
        gopher_quality = GopherQualityFilter(min_doc_words=10, max_doc_words=1000)
        self.check_filter(gopher_quality, get_doc("I am too small..."), "gopher_short_doc")
        self.check_filter(gopher_quality, get_doc("I am " * 20), "gopher_below_avg_threshold")
        self.check_filter(gopher_quality, get_doc("interconnection " * 20), "gopher_above_avg_threshold")
        self.check_filter(gopher_quality, get_doc("# comment " * 20), "gopher_too_many_hashes")
        self.check_filter(gopher_quality, get_doc("... comment " * 20), "gopher_too_many_ellipsis")
        text = "the ./!*?<><> apple <?////> orange  ++ interconnection !<>??? have" * 20
        self.check_filter(gopher_quality, get_doc(text), "gopher_below_alpha_threshold")
        self.assertTrue(gopher_quality(get_doc(TEXT_LF_1)))

    def test_lambda(self):
        doc = Document(text=TEXT_LF_1, id="0", metadata={"test": 1})
        lambda_filter = LambdaFilter(filter_function=lambda doc: doc.metadata["test"] > 0)
        self.assertTrue(lambda_filter.filter(doc))
        doc.metadata["test"] = -1
        self.assertFalse(lambda_filter.filter(doc))

    @require_fasttext
    def test_language(self):
        language_filter = LanguageFilter(languages=("en", "it"))

        doc1 = Document(text=TEXT_LF_1, id="0")
        self.assertTrue(language_filter.filter(doc1))
        self.assertEqual(doc1.metadata["language"], "en")

        doc2 = Document(text=TEXT_LF_2, id="0")
        self.assertFalse(language_filter.filter(doc2))
        self.assertEqual(doc2.metadata["language"], "fr")

        doc3 = Document(text=TEXT_LF_3, id="0")
        self.assertFalse(language_filter.filter(doc3))
        self.assertEqual(doc3.metadata["language"], "pt")

        doc4 = Document(text=TEXT_LF_4, id="0")
        self.assertTrue(language_filter.filter(doc4))
        self.assertEqual(doc4.metadata["language"], "it")

    def test_regex(self):
        regex_filter = RegexFilter(regex_exp=r"(?i)copyright")
        self.assertFalse(regex_filter.filter(get_doc(TEXT_LF_1 + "\n\nCoPyRiGhT")))
        self.assertTrue(regex_filter.filter(get_doc(TEXT_LF_1)))

    @require_nltk
    def test_unigram_prob(self):
        unigram_filter = UnigramLogProbFilter(logprobs_threshold=-10)
        self.assertTrue(unigram_filter.filter(Document(text=TEXT_LF_1, id="0")))
        self.assertFalse(unigram_filter.filter(Document(text="Cacophony Pareidolia Serendipity", id="0")))

    @require_tldextract
    def test_url(self):
        url_filter = URLFilter(extra_domains=("blocked.com", "danger.org", "badsubdomain.nice.com"))

        for url, result in (
            ("https://blocked.com/some-sub-url?with=stuff", "domain"),
            ("https://hey.danger.org/some-sub-url?with=stuff", "domain"),
            ("http://hey.danger.org/some-sub-url?with=stuff", "domain"),
            ("http://www.danger.org/some-sub-url?with=stuff", "domain"),
            ("https://nice.com/some-sub-url?with=stuff", True),
            ("https://badsubdomain.nice.com/some-sub-url?with=stuff", "subdomain"),
            ("https://sdsd.badsubdomain.nice.com/some-sub-url?with=stuff", True),
            ("https://blocke.dcom/some-sub-url?with=stuff", True),
        ):
            doc = get_doc(TEXT_LF_1, url)
            if result is True:
                assert url_filter.filter(doc)
            else:
                self.check_filter(url_filter, doc, result)


class TestSamplerFilter(unittest.TestCase):
    def test_rate_controls_keep_ratio(self):
        sampler = SamplerFilter(rate=0.5, seed=42)
        docs = [get_doc(f"text {i}") for i in range(1000)]
        kept = [d for d in docs if sampler.filter(d)]
        assert 400 < len(kept) < 600

    def test_deterministic_with_same_seed(self):
        sampler1 = SamplerFilter(rate=0.5, seed=123)
        sampler2 = SamplerFilter(rate=0.5, seed=123)
        docs = [get_doc(f"text {i}") for i in range(100)]
        results1 = [sampler1.filter(d) for d in docs]
        results2 = [sampler2.filter(d) for d in docs]
        assert results1 == results2


@require_nltk
class TestC4QualityFilter(unittest.TestCase):
    def setUp(self):
        self.c4_filter = C4QualityFilter()

    def test_keeps_quality_text(self):
        good_text = (
            "This is a well-written paragraph with several sentences.\n"
            "The content continues with more good information here.\n"
            "Another important point is being made in this line.\n"
            "We can see that this document has quality content.\n"
            "Finally, the conclusion wraps everything up nicely.\n"
            "An additional sentence to ensure we pass the threshold."
        )
        assert self.c4_filter.filter(get_doc(good_text)) is True

    def test_rejects_lorem_ipsum(self):
        text = (
            "This is a sentence with lorem ipsum dolor sit amet.\n"
            "Another sentence with proper punctuation.\n"
            "More content to fill out the document.\n"
            "Even more content is needed to pass the check.\n"
            "The final sentence of the document."
        )
        result, reason = self.c4_filter.filter(get_doc(text))
        assert result is False
        assert reason == "lorem_ipsum"

    def test_rejects_curly_bracket(self):
        text = (
            "This is some code with a { bracket.\n"
            "Another sentence follows this one.\n"
            "More content to fill out the document.\n"
            "Even more content is needed to pass.\n"
            "The final sentence of the document."
        )
        result, reason = self.c4_filter.filter(get_doc(text))
        assert result is False
        assert reason == "curly_bracket"

    def test_rejects_too_few_sentences(self):
        result, reason = self.c4_filter.filter(get_doc("Just one sentence."))
        assert result is False
        assert reason == "too_few_sentences"

    def test_removes_policy_lines_from_text(self):
        c4 = C4QualityFilter(min_num_sentences=-1, min_words_per_line=-1)
        d = get_doc("This site uses cookies for tracking.\nAnother valid line here today.")
        c4.filter(d)
        assert "cookies" not in d.text


class TestC4ParagraphFilter(unittest.TestCase):
    def test_keeps_doc_with_long_paragraphs(self):
        para = "x" * 250
        text = f"{para}\n{para}\n{para}"
        assert C4ParagraphFilter().filter(get_doc(text)) is True

    def test_rejects_short_paragraphs(self):
        result, reason = C4ParagraphFilter().filter(get_doc("short\nshort\nshort"))
        assert result is False
        assert "paragraphs" in reason


@require_nltk
class TestFineWebQualityFilter(unittest.TestCase):
    def test_rejects_empty_doc(self):
        result, reason = FineWebQualityFilter().filter(get_doc(""))
        assert result is False
        assert reason == "empty"

    def test_rejects_low_punctuation_ratio(self):
        fw = FineWebQualityFilter(line_punct_thr=0.5)
        text = "No punctuation here\nAnother line without any\nStill nothing"
        result, reason = fw.filter(get_doc(text))
        assert result is False
        assert reason == "line_punct_ratio"

    def test_rejects_mostly_short_lines(self):
        fw = FineWebQualityFilter(short_line_thr=0.5, short_line_length=30, line_punct_thr=0.0)
        result, reason = fw.filter(get_doc("hi.\nbye.\nok.\nno.\nyes."))
        assert result is False
        assert reason == "short_line_ratio"

    def test_keeps_quality_text(self):
        fw = FineWebQualityFilter()
        text = (
            "This is a well-written paragraph that ends with proper punctuation.\n"
            "The content here is substantial and has meaningful information.\n"
            "Quality filtering ensures that only the best documents pass through.\n"
            "Each line here is long enough and ends with a terminal punctuation mark.\n"
            "This document should pass all the quality checks applied by the filter."
        )
        assert fw.filter(get_doc(text)) is True

    def test_rejects_list_heavy_doc(self):
        fw = FineWebQualityFilter(new_line_ratio=0.1)
        result, reason = fw.filter(get_doc("\n".join(["a"] * 50)))
        assert result is False


class TestBaseFilterRunPipeline(unittest.TestCase):
    """Test BaseFilter.run() stat tracking via SamplerFilter."""

    def test_run_tracks_forwarded_and_dropped_stats(self):
        sampler_keep = SamplerFilter(rate=1.0, seed=0)
        sampler_drop = SamplerFilter(rate=0.0, seed=0)
        docs = [get_doc(f"text {i}") for i in range(3)]

        list(sampler_keep.run(iter(docs)))
        assert sampler_keep.stats["total"].total == 3
        assert sampler_keep.stats["forwarded"].total == 3

        list(sampler_drop.run(iter(docs)))
        assert sampler_drop.stats["total"].total == 3
        assert sampler_drop.stats["dropped"].total == 3
