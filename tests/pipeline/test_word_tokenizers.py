import unittest

from datatrove.utils.word_tokenizers import TibetanTokenizer, load_tokenizer_assignments, load_word_tokenizer


SAMPLE_TEXT = (
    "'I wish it need not have happened in my time,' said Frodo. 'So do I,' said Gandalf, 'and so do all who live to "
    "see such times. But that is not for them to decide. All we have to decide is what to do with the time that is "
    "given us.' Hello world! \n\n ქართული \n\t Hello\nworld! "
)


def get_unique_tokenizers():
    uniq_toks = set()
    for language in load_tokenizer_assignments().keys():
        tokenizer = load_word_tokenizer(language)
        if (tokenizer.__class__, tokenizer.language) in uniq_toks:
            continue
        uniq_toks.add((tokenizer.__class__, tokenizer.language))
        yield language, tokenizer


class TestWordTokenizers(unittest.TestCase):
    def test_word_tokenizers(self):
        for language, tokenizer in get_unique_tokenizers():
            tokens = tokenizer.word_tokenize(SAMPLE_TEXT)
            assert len(tokens) >= 1, f"'{language}' tokenizer doesn't output tokens"
            is_stripped = [token == token.strip() for token in tokens]
            assert all(is_stripped), f"'{language}' tokenizer tokens contain whitespaces"

    def test_sent_tokenizers(self):
        for language, tokenizer in get_unique_tokenizers():
            sents = tokenizer.sent_tokenize(SAMPLE_TEXT)
            assert len(sents) >= 1, f"'{language}' tokenizer doesn't output sentences"
            is_stripped = [sent == sent.strip() for sent in sents]
            assert all(is_stripped), f"'{language}' tokenizer sentences contain whitespaces"

    def test_span_tokenizers(self):
        for language, tokenizer in get_unique_tokenizers():
            sents = tokenizer.sent_tokenize(SAMPLE_TEXT)
            spans = tokenizer.span_tokenize(SAMPLE_TEXT)
            assert len(spans) >= 1, f"'{language}' tokenizer doesn't output spans"
            spans_match_sents = [sent in SAMPLE_TEXT[span[0] : span[1]] for sent, span in zip(sents, spans)]
            assert (tokenizer.language == "ur" or isinstance(tokenizer, TibetanTokenizer)) or all(spans_match_sents), (
                f"'{language}' tokenizer spans don't match with sentences"
            )

    def test_english_tokenizer(self):
        en_tokenizer = load_word_tokenizer("en")
        tokenizer_words = en_tokenizer.word_tokenize(SAMPLE_TEXT)

        self.assertEqual(
            [
                "'",
                "I",
                "wish",
                "it",
                "need",
                "not",
                "have",
                "happened",
                "in",
                "my",
                "time",
                ",",
                "'",
                "said",
                "Frodo",
                ".",
                "'",
                "So",
                "do",
                "I",
                ",",
                "'",
                "said",
                "Gandalf",
                ",",
                "'",
                "and",
                "so",
                "do",
                "all",
                "who",
                "live",
                "to",
                "see",
                "such",
                "times",
                ".",
                "But",
                "that",
                "is",
                "not",
                "for",
                "them",
                "to",
                "decide",
                ".",
                "All",
                "we",
                "have",
                "to",
                "decide",
                "is",
                "what",
                "to",
                "do",
                "with",
                "the",
                "time",
                "that",
                "is",
                "given",
                "us",
                ".",
                "'",
                "Hello",
                "world",
                "!",
                "ქართული",
                "Hello",
                "world",
                "!",
            ],
            tokenizer_words,
            "SpaCy tokenizer and multilingual tokenizer differ",
        )
