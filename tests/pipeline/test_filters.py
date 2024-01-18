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


def get_doc(text, url=None):
    return Document(text, id="0", metadata={"url": url})


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

        self.assertTrue(language_filter.filter(Document(text=TEXT_LF_1, id="0")))
        self.assertFalse(language_filter.filter(Document(text=TEXT_LF_2, id="0")))
        self.assertFalse(language_filter.filter(Document(text=TEXT_LF_3, id="0")))
        self.assertTrue(language_filter.filter(Document(text=TEXT_LF_4, id="0")))

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
