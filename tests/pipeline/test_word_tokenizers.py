import unittest

from nltk.tokenize import word_tokenize

from datatrove.tools.word_tokenizers import default_tokenizer
from datatrove.utils.typeshelper import Languages


SAMPLE_TEXT = (
    "I wish it need not have happened in my time,' said Frodo. 'So do I,' said Gandalf, 'and so do all who live to "
    "see such times. But that is not for them to decide. All we have to decide is what to do with the time that is "
    "given us.' Hello world! \n\n ქართული \n\t Hello\nworld! "
)


class TestWordTokenizers(unittest.TestCase):
    def test_word_tokenizers(self):
        for language in default_tokenizer.languages:
            tokens = default_tokenizer.word_tokenize(SAMPLE_TEXT, language)
            assert len(tokens) >= 1, f"'{language}' tokenizer doesn't output tokens"
            is_stripped = [token == token.strip() for token in tokens]
            assert all(is_stripped), f"'{language}' tokenizer tokens contain whitespaces"

    def test_sent_tokenizers(self):
        for language in default_tokenizer.languages:
            sents = default_tokenizer.sent_tokenize(SAMPLE_TEXT, language)
            assert len(sents) >= 1, f"'{language}' tokenizer doesn't output sentences"
            is_stripped = [sent == sent.strip() for sent in sents]
            assert all(is_stripped), f"'{language}' tokenizer sentences contain whitespaces"

    def test_english_tokenizer(self):
        nltk_words = word_tokenize(SAMPLE_TEXT, language="english")
        tokenizer_words = default_tokenizer.word_tokenize(SAMPLE_TEXT, language=Languages.english)

        self.assertEqual(nltk_words, tokenizer_words, "NLTK tokenizer and multilingual tokenizer differ")
