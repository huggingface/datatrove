import unittest

from nltk.tokenize import word_tokenize

from datatrove.utils.word_tokenizers import WORD_TOKENIZER_FACTORY, load_word_tokenizer


SAMPLE_TEXT = (
    "I wish it need not have happened in my time,' said Frodo. 'So do I,' said Gandalf, 'and so do all who live to "
    "see such times. But that is not for them to decide. All we have to decide is what to do with the time that is "
    "given us.' Hello world! \n\n ქართული \n\t Hello\nworld! "
)


class TestWordTokenizers(unittest.TestCase):
    def test_word_tokenizers(self):
        for language in WORD_TOKENIZER_FACTORY.keys():
            tokenizer = load_word_tokenizer(language)
            tokens = tokenizer.word_tokenize(SAMPLE_TEXT)
            assert len(tokens) >= 1, f"'{language}' tokenizer doesn't output tokens"
            is_stripped = [token == token.strip() for token in tokens]
            assert all(is_stripped), f"'{language}' tokenizer tokens contain whitespaces"

    def test_sent_tokenizers(self):
        for language in WORD_TOKENIZER_FACTORY.keys():
            tokenizer = load_word_tokenizer(language)
            sents = tokenizer.sent_tokenize(SAMPLE_TEXT)
            assert len(sents) >= 1, f"'{language}' tokenizer doesn't output sentences"
            is_stripped = [sent == sent.strip() for sent in sents]
            assert all(is_stripped), f"'{language}' tokenizer sentences contain whitespaces"

    def test_span_tokenizers(self):
        for language in WORD_TOKENIZER_FACTORY.keys():
            tokenizer = load_word_tokenizer(language)
            sents = tokenizer.sent_tokenize(SAMPLE_TEXT)
            spans = tokenizer.span_tokenize(SAMPLE_TEXT)
            assert len(spans) >= 1, f"'{language}' tokenizer doesn't output spans"
            spans_match_sents = [sent in SAMPLE_TEXT[span[0] : span[1]] for sent, span in zip(sents, spans)]
            assert all(spans_match_sents), f"'{language}' tokenizer spans don't match with sentences"

    def test_english_tokenizer(self):
        nltk_words = word_tokenize(SAMPLE_TEXT, language="english")

        en_tokenizer = load_word_tokenizer("en")
        tokenizer_words = en_tokenizer.word_tokenize(SAMPLE_TEXT)

        self.assertEqual(nltk_words, tokenizer_words, "NLTK tokenizer and multilingual tokenizer differ")
