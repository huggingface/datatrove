import unittest

from src.datatrove.utils.text import PUNCTUATION, TextNormConfig, simplify_text


class TestTextTransformation(unittest.TestCase):
    def test_text_table_norm(self):
        text = "|$17.56||1|\n|$15.37||2599|"
        config = TextNormConfig(norm_numbers=True, remove_punctuation=True, norm_whitespace=True)
        transformed_text = simplify_text(text, config)
        expected_text = "0 0 0 0"
        self.assertEqual(transformed_text, expected_text)

    def test_punc_normalization(self):
        text = PUNCTUATION
        config = TextNormConfig(remove_punctuation=True)
        transformed_text = simplify_text(text, config)
        # Should be just 0, because there is a strange 1 in special symbols
        expected_text = "0"
        self.assertEqual(transformed_text, expected_text)
