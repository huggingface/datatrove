import csv
import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache, partial
from typing import Callable, Iterator

import regex
from loguru import logger

from datatrove.utils._import_utils import ASSETS_PATH, check_required_dependencies
from datatrove.utils.text import TERMINAL_PUNCTUATION


def strip_strings(els: list[str]) -> list[str]:
    return [el.strip() for el in els if len(el.strip()) > 0]


def simple_span_tokenize(text: str, sents: list[str]) -> Iterator[tuple[int, int]]:
    if len(sents) == 1:
        yield 0, len(text)
        return
    start_index = 0
    for sent in sents:
        start_char = text.index(sent, start_index)
        end_char = start_char + len(sent)
        start_index = end_char
        yield start_char, end_char


# https://github.com/explosion/spaCy/issues/13207
def chunk_text_on_bytes(text: str, max_chunk_size: int = 1_000_000):
    def __utf8len(s: str):
        return len(s.encode("utf-8"))

    factor = len(text) / __utf8len(text) if __utf8len(text) > 0 else 1
    increase_by = int(max(min(max_chunk_size * 0.1, 10), 1))
    initial_size_guess = int(max(max_chunk_size * factor - 10, 1))
    final_list = []
    remaining = text
    while len(remaining):
        part = remaining[:initial_size_guess]
        if __utf8len(part) > max_chunk_size:
            initial_size_guess = max(initial_size_guess - min(max_chunk_size * 0.001, 10), 1)
            continue
        cut_after = initial_size_guess
        while __utf8len(part) < max_chunk_size and part != remaining:
            cut_after = min(len(remaining), cut_after + increase_by)
            part = remaining[:cut_after]

        if __utf8len(part) > max_chunk_size:
            cut_after -= increase_by
        final_list.append(remaining[:cut_after])
        remaining = remaining[cut_after:]

    return final_list


class WordTokenizer(ABC):
    def __init__(self, language: str | None = None):
        self.language = language

    @abstractmethod
    def word_tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def sent_tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        pass


class NLTKTokenizer(WordTokenizer):
    def __init__(self, language: str):
        super().__init__(language)
        check_required_dependencies(f"{language} word tokenizer", ["nltk"])
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from nltk import load

            self._tokenizer = load(f"tokenizers/punkt/{self.language}.pickle")
        return self._tokenizer

    def word_tokenize(self, text) -> list[str]:
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(text, language=self.language)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from nltk.tokenize import sent_tokenize

        sents = sent_tokenize(text, language=self.language)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return list(self.tokenizer.span_tokenize(text))


class SpaCyTokenizer(WordTokenizer):
    def __init__(self, language: str, config=None):
        super().__init__(language)
        check_required_dependencies(f"{language} word tokenizer", ["spacy"])
        if language == "vi":
            check_required_dependencies(f"{language} word tokenizer", ["pyvi"])
        elif language == "zh":
            config = {"nlp": {"tokenizer": {"segmenter": "jieba"}}}
            check_required_dependencies(f"{language} word tokenizer", ["jieba"])
        self.config = config
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import spacy

            # Important to hot-fix the memory leak in Japanese Tokenizer
            from datatrove.utils.japanese_tokenizer import JapaneseTokenizer  # noqa: F401

            if self.config is None:
                self._tokenizer = spacy.blank(self.language)
            else:
                self._tokenizer = spacy.blank(self.language, config=self.config)
            self._tokenizer.add_pipe("sentencizer")
        return self._tokenizer

    def _do_tokenize(self, text: str):
        # japanese has a max byte length
        texts = [text] if self.language != "ja" else chunk_text_on_bytes(text, 40000)
        self.tokenizer.max_length = len(text)
        try:
            return [self.tokenizer(t, disable=["parser", "tagger", "ner"]) for t in texts]
        except Exception as e:
            # this dumb string breaks the tokenizer completely
            if "IS_ALPHA" in text:
                return [self.tokenizer(t.replace("IS_ALPHA", ""), disable=["parser", "tagger", "ner"]) for t in texts]
            else:
                raise e

    def word_tokenize(self, text: str) -> list[str]:
        # Make sure to do all the token processing inside the memory zone, as after that memory address to tokens
        # are not longer valid
        with self.tokenizer.memory_zone():
            self.tokenizer.max_length = len(text) + 10
            tokens = [token.text for tok_chunk in self._do_tokenize(text) for token in tok_chunk]
            return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        with self.tokenizer.memory_zone():
            self.tokenizer.max_length = len(text) + 10
            sents = [sent.text for t in self._do_tokenize(text) for sent in t.sents]
            return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        spans = []
        with self.tokenizer.memory_zone():
            for tok_text in self._do_tokenize(text):
                start = spans[-1][1] if spans else 0
                for sent in tok_text.sents:
                    spans.append((start + sent.start_char, start + sent.end_char))
        return spans


class StanzaTokenizer(WordTokenizer):
    def __init__(self, language: str, **stanza_kwargs):
        super().__init__(language)
        check_required_dependencies(f"{language} word tokenizer", ["stanza"])
        self.stanza_kwargs = stanza_kwargs
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import stanza
            from stanza.pipeline.core import DownloadMethod

            self._tokenizer = stanza.Pipeline(
                self.language,
                processors="tokenize",
                download_method=DownloadMethod.REUSE_RESOURCES,
                **self.stanza_kwargs,
            )

        return self._tokenizer

    def word_tokenize(self, text: str) -> list[str]:
        doc = self.tokenizer(text)
        tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        doc = self.tokenizer(text)
        sents = [sentence.text for sentence in doc.sentences]
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        doc = self.tokenizer(text)
        return [(sent.tokens[0].start_char, sent.tokens[-1].end_char) for sent in doc.sentences]


class ThaiTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("th word tokenizer", ["pythainlp"])

    def word_tokenize(self, text: str) -> list[str]:
        from pythainlp.tokenize import word_tokenize as th_word_tokenize

        tokens = th_word_tokenize(text, keep_whitespace=False, engine="newmm-safe")
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from pythainlp.tokenize import sent_tokenize as th_sent_tokenize

        sents = th_sent_tokenize(text)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class IndicNLPTokenizer(WordTokenizer):
    def __init__(self, language: str):
        super().__init__(language)
        check_required_dependencies(f"{language} word tokenizer", [("indicnlp", "indic-nlp-library")])

    def word_tokenize(self, text) -> list[str]:
        from indicnlp.tokenize.indic_tokenize import trivial_tokenize as indicnlp_trivial_tokenize

        tokens = indicnlp_trivial_tokenize(text, self.language)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from indicnlp.tokenize.sentence_tokenize import sentence_split

        sents = sentence_split(text, lang=self.language)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class KiwiTokenizer(WordTokenizer):
    def __init__(self, model_type="sbg"):
        super().__init__()
        check_required_dependencies("ko word tokenizer", ["kiwipiepy"])
        self.model_type = model_type
        self._tokenizer = None
        self._preprocess_regex = re.compile("[0-9,]{20,}")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from kiwipiepy import Kiwi

            self._tokenizer = Kiwi(model_type=self.model_type)
        return self._tokenizer

    def preprocess(self, text):
        # seems to have issue with very large numbers
        return self._preprocess_regex.sub("", text)

    def word_tokenize(self, text: str) -> list[str]:
        tokens = [text[token.start : token.end] for token in self.tokenizer.tokenize(self.preprocess(text))]
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        sents = [sent.text for sent in self.tokenizer.split_into_sents(self.preprocess(text))]
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return [(sent.start, sent.end) for sent in self.tokenizer.split_into_sents(self.preprocess(text))]


class KhmerTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("khmer word tokenizer", [("khmernltk", "khmer-nltk")])

    def word_tokenize(self, text: str) -> list[str]:
        from khmernltk import word_tokenize

        tokens = word_tokenize(text, return_tokens=True)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from khmernltk import sentence_tokenize

        return strip_strings(sentence_tokenize(text))

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class LaoTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("laos word tokenizer", ["laonlp"])

    def word_tokenize(self, text: str) -> list[str]:
        from laonlp.tokenize import word_tokenize

        tokens = word_tokenize(text)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from laonlp.tokenize import sent_tokenize

        return strip_strings(sent_tokenize(text))

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class TibetanTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("tibetan word tokenizer", ["botok"])
        self._wt = None
        self._whitespace_regex = re.compile(r"\s+")

    @property
    def wt(self):
        if self._wt is None:
            from botok import WordTokenizer

            self._wt = WordTokenizer()
        return self._wt

    def _try_tokenize(self, text: str) -> list[str]:
        try:
            return self.wt.tokenize(text, split_affixes=False)
        except Exception as e:
            logger.warning(f"Failed to tokenize with botok: {e}. Trying without spaces...")
            return self.wt.tokenize(self._whitespace_regex.sub("", text), split_affixes=False)

    def word_tokenize(self, text: str) -> list[str]:
        return strip_strings([tok.text for tok in self._try_tokenize(text)])

    def sent_tokenize(self, text: str) -> list[str]:
        from botok.tokenizers.sentencetokenizer import sentence_tokenizer

        tokens = self._try_tokenize(text)
        sents = sentence_tokenizer(tokens)
        out = ["".join([word.text for word in s["tokens"]]) for s in sents]
        return strip_strings(out)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        from botok.tokenizers.sentencetokenizer import get_sentence_indices

        tokens = self._try_tokenize(text)
        idxs = get_sentence_indices(tokens)
        return [(sentence["start"], sentence["end"] + 1) for sentence in idxs]


class WhitespaceTokenizer(WordTokenizer):
    """
    This is a fallback tokenizer when no other tokenizer is available.
    """

    def __init__(self):
        super().__init__()
        # should not split on acronyms "(?:\p{{Lu}}\.)"
        self._sent_regex = regex.compile(
            rf"(?:(?:\p{{Lu}}\.)|.)+?[{re.escape(''.join(TERMINAL_PUNCTUATION))}\n]+[\"'”]?", regex.UNICODE
        )

    @property
    @lru_cache(1)
    def _spacy_xx(self):
        # works generally well for white spaces, but does not work to split sentences with a different script
        return SpaCyTokenizer("xx")

    def word_tokenize(self, text) -> list[str]:
        return self._spacy_xx.word_tokenize(text)

    def sent_tokenize(self, text: str) -> list[str]:
        sents = self._sent_regex.findall(text)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class BurmeseTokenizer(WhitespaceTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("burmese word tokenizer", [("pyidaungsu", "pyidaungsu-numpy2")])
        self._wt = None

    def word_tokenize(self, text: str) -> list[str]:
        import pyidaungsu as pds

        tokens = pds.tokenize(text, form="word")
        return strip_strings(tokens)


"""
    The actual tokenizer assignments are saved in src/datatrove/assets/tokenizer_assignments.csv
    If you know a better tokenizer or better proxy language, please submit a PR
"""


@lru_cache(maxsize=1)
def load_tokenizer_assignments() -> dict[str, Callable[[], WordTokenizer]]:
    def tok_factory_wrapper(class_name, arg):
        if class_name == "SpaCyTokenizer":
            tok_class = SpaCyTokenizer
        elif class_name == "StanzaTokenizer":
            tok_class = StanzaTokenizer
        elif class_name == "ThaiTokenizer":
            tok_class = ThaiTokenizer
        elif class_name == "IndicNLPTokenizer":
            tok_class = IndicNLPTokenizer
        elif class_name == "KiwiTokenizer":
            tok_class = KiwiTokenizer
        elif class_name == "KhmerTokenizer":
            tok_class = KhmerTokenizer
        elif class_name == "LaoTokenizer":
            tok_class = LaoTokenizer
        elif class_name == "TibetanTokenizer":
            tok_class = TibetanTokenizer
        elif class_name == "BurmeseTokenizer":
            tok_class = BurmeseTokenizer
        elif class_name == "WhitespaceTokenizer":
            tok_class = WhitespaceTokenizer
        else:
            raise ValueError(f'Invalid tokenizer class "{class_name}"')

        if arg:
            return tok_class(arg)
        return tok_class()

    word_tokenizer_factories = {}
    with open(os.path.join(ASSETS_PATH, "tokenizer_assignment.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            code_3, code_1, script, tok_class_name, tok_code, default_script, default_code_1 = (
                row["code_3"],
                row["code_1"],
                row["script"],
                row["type"],
                row["tok_code"],
                row["default_script"],
                row["default_code_1"],
            )

            if not tok_class_name:
                continue

            tok_factory = partial(tok_factory_wrapper, tok_class_name, tok_code)

            code_3_script = f"{code_3}_{script}"
            if code_3_script not in word_tokenizer_factories:
                word_tokenizer_factories[code_3_script] = tok_factory
                if default_script:
                    word_tokenizer_factories[code_3] = tok_factory
            code_1_script = f"{code_1}_{script}"
            if code_1 and default_code_1 and code_1_script not in word_tokenizer_factories:
                word_tokenizer_factories[code_1_script] = tok_factory
                if default_script:
                    word_tokenizer_factories[code_1] = tok_factory

    return word_tokenizer_factories


@lru_cache(maxsize=None)
def load_word_tokenizer(language_or_tok: str | WordTokenizer) -> WordTokenizer:
    if isinstance(language_or_tok, WordTokenizer):
        # for custom tokenizers
        return language_or_tok
    word_tokenizer_factories = load_tokenizer_assignments()
    if language_or_tok not in word_tokenizer_factories:
        raise ValueError(
            f"Language '{language_or_tok}' doesn't have a tokenizer assigned. Pass in a "
            f"WordTokenizer directly or update tokenizer_assignment.csv"
        )
    return word_tokenizer_factories[language_or_tok]()
