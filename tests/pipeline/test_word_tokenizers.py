import unittest

from datatrove.utils.word_tokenizers import (
    KiwiTokenizer,
    TibetanTokenizer,
    load_tokenizer_assignments,
    load_word_tokenizer,
)


SAMPLE_TEXT = (
    "'I wish it need not have happened in my time,' said Frodo. 'So do I,' said Gandalf, 'and so do all who live to "
    "see such times. But that is not for them to decide. All we have to decide is what to do with the time that is "
    "given us.' Hello world! \n\n ქართული \n\t Hello\nworld! "
)


def get_unique_tokenizers():
    uniq_toks = set()
    for language in load_tokenizer_assignments().keys():
        print(f"[LOAD] Loading tokenizer for language: {language}")
        try:
            tokenizer = load_word_tokenizer(language)
            print(
                f"[LOAD] Successfully loaded tokenizer for language: {language}, type: {tokenizer.__class__.__name__}"
            )

            # Skip KiwiTokenizer (Korean) - it has missing model files in CI
            if isinstance(tokenizer, KiwiTokenizer):
                print(f"[SKIP] Skipping KiwiTokenizer for language: {language} (missing model files in CI)")
                continue
        except Exception as e:
            print(f"[ERROR] ERROR loading tokenizer for language {language}: {e}")
            import traceback

            traceback.print_exc()
            raise
        if (tokenizer.__class__, tokenizer.language) in uniq_toks:
            print(f"[SKIP] Skipping duplicate tokenizer for language: {language}")
            continue
        uniq_toks.add((tokenizer.__class__, tokenizer.language))
        print(f"[YIELD] Yielding tokenizer for language: {language}, type: {tokenizer.__class__.__name__}")
        yield language, tokenizer
        print(f"[YIELD] Returned from yield for language: {language}")


class TestWordTokenizers(unittest.TestCase):
    def test_word_tokenizers(self):
        for language, tokenizer in get_unique_tokenizers():
            print(f"[TEST] Testing word_tokenize for language: {language}, tokenizer: {tokenizer.__class__.__name__}")

            # Add detailed logging for SpaCyTokenizer lazy loading
            if tokenizer.__class__.__name__ == "SpaCyTokenizer":
                print(f"[TEST] SpaCyTokenizer detected, language code: {tokenizer.language}")
                print("[TEST] About to access tokenizer property (lazy loading spaCy model)...")

            print(f"[TEST] About to call word_tokenize for {language}...")
            try:
                tokens = tokenizer.word_tokenize(SAMPLE_TEXT)
                print(f"[TEST] Successfully called word_tokenize for {language}, got {len(tokens)} tokens")
            except Exception as e:
                print(f"[TEST] Exception calling word_tokenize for {language}: {e}")
                import traceback

                traceback.print_exc()
                raise
            assert len(tokens) >= 1, f"'{language}' tokenizer doesn't output tokens"
            is_stripped = [token == token.strip() for token in tokens]
            assert all(is_stripped), f"'{language}' tokenizer tokens contain whitespaces"

    def test_sent_tokenizers(self):
        for language, tokenizer in get_unique_tokenizers():
            print(f"[TEST] Testing sent_tokenize for language: {language}, tokenizer: {tokenizer.__class__.__name__}")

            # Add detailed logging for SpaCyTokenizer lazy loading
            if tokenizer.__class__.__name__ == "SpaCyTokenizer":
                print(f"[TEST] SpaCyTokenizer detected, language code: {tokenizer.language}")
                print("[TEST] About to access tokenizer property (lazy loading spaCy model)...")

            print(f"[TEST] About to call sent_tokenize for {language}...")
            try:
                sents = tokenizer.sent_tokenize(SAMPLE_TEXT)
                print(f"[TEST] Successfully called sent_tokenize for {language}, got {len(sents)} sentences")
            except Exception as e:
                print(f"[TEST] Exception calling sent_tokenize for {language}: {e}")
                import traceback

                traceback.print_exc()
                raise
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
