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
    return Document(text, data_id="0", metadata={"url": url})


class TestFilters(unittest.TestCase):
    def check_filter(self, filter, doc, filter_reason):
        filter_result = filter.filter(doc)
        self.assertEqual(type(filter_result), tuple)
        self.assertEqual(filter_result[1], filter_reason)

    def test_gopher_repetition(self):
        gopher_repetition = GopherRepetitionFilter()

        self.check_filter(gopher_repetition, get_doc("I am your father.\n" * 4), "dup_line_frac")
        self.check_filter(gopher_repetition, get_doc("I am your father.\n\n" * 4), "dup_para_frac")
        text = "I am groot.\n\n" + "You are a wizard.\n\n" + "I am your father.\n\n" * 2 + f"{'x' * 30}.\n\n" * 2
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
        self.check_filter(
            gopher_quality, get_doc("./!*?<><> apple orange interconnection " * 20), "gopher_below_alpha_threshold"
        )

    def test_lambda(self):
        doc = Document(content=TEXT_LF_1, data_id="0", metadata={"test": 1})
        lambda_filter = LambdaFilter(filter_function=lambda doc: doc.metadata["test"] > 0)
        self.assertTrue(lambda_filter.filter(doc))
        doc.metadata["test"] = -1
        self.assertFalse(lambda_filter.filter(doc))

    def test_language(self):
        language_filter = LanguageFilter(languages=("en", "it"))

        self.assertTrue(language_filter.filter(Document(content=TEXT_LF_1, data_id="0")))
        self.assertFalse(language_filter.filter(Document(content=TEXT_LF_2, data_id="0")))
        self.assertFalse(language_filter.filter(Document(content=TEXT_LF_3, data_id="0")))
        self.assertTrue(language_filter.filter(Document(content=TEXT_LF_4, data_id="0")))

    def test_regex(self):
        regex_filter = RegexFilter(regex_exp=r"(?i)copyright")
        self.assertFalse(regex_filter.filter(get_doc(TEXT_LF_1 + "\n\nCoPyRiGhT")))
        self.assertTrue(regex_filter.filter(get_doc(TEXT_LF_1)))

    def test_unigram_prob(self):
        unigram_filter = UnigramLogProbFilter(logprobs_threshold=-10)
        self.assertTrue(unigram_filter.filter(Document(content=TEXT_LF_1, data_id="0")))
        self.assertFalse(unigram_filter.filter(Document(content="Cacophony Pareidolia Serendipity", data_id="0")))

    def test_url(self):
        def get_element(s):
            for e in s:
                return e

        url_filter = URLFilter()

        url = get_element(url_filter.block_listed_domains)
        self.check_filter(url_filter, get_doc(TEXT_LF_1, url), "domain")
