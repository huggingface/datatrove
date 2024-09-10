import unittest

from datatrove.data import Document
from datatrove.pipeline.enrishers import (
    C4QualityEnrisher,
    GopherQualityEnrisher,
)


def get_doc(text, url=None):
    return Document(text, id="0", metadata={"url": url})


class TestEnrishers(unittest.TestCase):
    def test_c4_quality(self):
        self.maxDiff = None
        c4_quality = C4QualityEnrisher(store_lines=True)
        doc = get_doc("I am groot{.\nYou are a javascript wizard use cookies")
        output_doc = c4_quality.enrish(doc)
        expected_metadata = {
            "lines": [
                {
                    "max_word_length": 7,
                    "has_curly_bracket": True,
                    "has_javascript": False,
                    "has_lorem_ipsum": False,
                    "has_policy": False,
                    "no_terminal_punct": False,
                    "words_per_line": 3,
                    "line": "I am groot{.",
                },
                {
                    "max_word_length": 10,
                    "has_curly_bracket": False,
                    "has_javascript": True,
                    "has_lorem_ipsum": False,
                    "has_policy": True,
                    "no_terminal_punct": True,
                    "words_per_line": 7,
                    "line": "You are a javascript wizard use cookies",
                },
            ],
            "num_sentences": 2,
        }
        self.assertDictEqual(output_doc.metadata["c4_quality"], expected_metadata)

    def test_gopher(self):
        self.maxDiff = None
        gopher_quality = GopherQualityEnrisher()

        doc = get_doc("I am too small...")
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["n_non_symbol_words"], 4)

        doc = get_doc("I am " * 20)
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["avg_word_length"], 1.5)

        doc = get_doc("# comment " * 20)
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["hash_to_word_ratio"], 0.5)

        doc = get_doc("... comment " * 20)
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["ellipsis_to_word_ratio"], 0.5)

        doc = get_doc("I am Frank:\n - I am the Frank\n - I am Frank\n - I am Frank")
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["bullet_lines_ratio"], 0.75)

        doc = get_doc("I am Frank...\nI am Frank...\nI am Frank...\nI am Frank...")
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["end_ellipsis_ratio"], 1.0)

        text = "the ./!*?<><> apple <?////> orange /!. interconnection have"
        doc = get_doc(text)
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["non_alpha_words_ratio"], 0.25)

        doc = get_doc("The name is Frank " * 20)
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["stop_words_count"], 20)

        doc = get_doc("I am Frank\n\nI am Frank\n\nI am Frank\n\nI am Frank")
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["dup_para_frac"], 0.75)

        doc = get_doc("I am Frank\n\nI am Frank")
        output_doc = gopher_quality.enrish(doc)
        self.assertAlmostEqual(output_doc.metadata["gopher"]["dup_para_char_frac"], 0.454545, places=5)

        doc = get_doc("I am Frank\nI am Frank")
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["dup_line_frac"], 0.5)

        doc = get_doc("I am Frank\nI am Frank")
        output_doc = gopher_quality.enrish(doc)
        self.assertAlmostEqual(output_doc.metadata["gopher"]["dup_line_char_frac"], 0.47619, places=5)

        doc = get_doc("I am Frank, I am Frank, I am Frank")
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["top_2_gram"], 4 * 3 / len(doc.text))

        doc = get_doc("I am Frank, you are Jhon. I am Frank. I am Frank you are Jhon")
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(output_doc.metadata["gopher"]["top_3_gram"], 10 * 3 / len(doc.text))

        doc = get_doc("I am a solo traveller " * 4)
        output_doc = gopher_quality.enrish(doc)
        self.assertEqual(
            output_doc.metadata["gopher"]["duplicated_5_n_grams"],
            17 * 3 / len(doc.text),
        )
