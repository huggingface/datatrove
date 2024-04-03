import unittest

from nltk.tokenize import word_tokenize

from datatrove.tools.word_tokenizers import WORD_TOKENIZERS, get_word_tokenizer
from datatrove.utils.typeshelper import Languages


SAMPLE_TEXT = (
    "I wish it need not have happened in my time,' said Frodo. 'So do I,' said Gandalf, 'and so do all who live to "
    "see such times. But that is not for them to decide. All we have to decide is what to do with the time that is "
    "given us.'"
)


class TestWordTokenizers(unittest.TestCase):
    def test_word_tokenizers(self):
        for language in WORD_TOKENIZERS:
            tokenizer = WORD_TOKENIZERS[language]
            tokens = tokenizer.tokenize(SAMPLE_TEXT)
            assert len(tokens) >= 1, f"'{language}' tokenizer assertion failed"
            is_stripped = [token == token.strip() for token in tokens]
            assert all(is_stripped), f"'{language}' tokens contain whitespaces"

    def test_english_tokenizer(self):
        nltk_words = word_tokenize(SAMPLE_TEXT, language="english")

        tokenizer = get_word_tokenizer(Languages.english)
        tokenizer_words = tokenizer.tokenize(SAMPLE_TEXT)

        self.assertEqual(nltk_words, tokenizer_words, "NLTK tokenizer and multilingual tokenizer differ")
