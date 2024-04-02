import unittest

from datatrove.tools.word_tokenizers import WORD_TOKENIZERS


sample_text = "Hello world! \n\n ქართული \n\t Hello\nworld! " * 50


class TestWordTokenizers(unittest.TestCase):
    def test_word_tokenizers(self):
        for language in WORD_TOKENIZERS:
            tokenizer = WORD_TOKENIZERS[language]
            tokens = tokenizer.tokenize(sample_text)
            assert len(tokens) >= 1, f"'{language}' tokenizer assertion failed"
            is_stripped = [token == token.strip() for token in tokens]
            assert all(is_stripped), f"'{language}' tokens contain whitespaces"
