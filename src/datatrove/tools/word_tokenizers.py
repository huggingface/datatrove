from abc import ABC, abstractmethod
from typing import Callable

from datatrove.utils.typeshelper import Languages


def strip_strings(els: list[str]) -> list[str]:
    return [el.strip() for el in els if len(el.strip()) > 0]


class WordTokenizer(ABC):
    @abstractmethod
    def word_tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def sent_tokenize(self, text: str) -> list[str]:
        pass


class NLTKTokenizer(WordTokenizer):
    def __init__(self, punkt_language: str):
        super().__init__()
        self.punkt_language = punkt_language

    def word_tokenize(self, text) -> list[str]:
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(text, language=self.punkt_language)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from nltk.tokenize import sent_tokenize

        sents = sent_tokenize(text, language=self.punkt_language)
        return strip_strings(sents)


class SpaCyTokenizer(WordTokenizer):
    def __init__(self, spacy_language: str, config=None):
        super().__init__()
        import spacy

        if config is None:
            self.tokenizer = spacy.blank(spacy_language)
        else:
            self.tokenizer = spacy.blank(spacy_language, config=config)
        self.tokenizer.add_pipe("sentencizer")

    def word_tokenize(self, text: str) -> list[str]:
        self.tokenizer.max_length = len(text) + 10
        tokens = [token.text for token in self.tokenizer(text, disable=["parser", "tagger", "ner"])]
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        self.tokenizer.max_length = len(text) + 10
        sents = [sent.text for sent in self.tokenizer(text, disable=["parser", "tagger", "ner"]).sents]
        return strip_strings(sents)


class MultilingualTokenizer:
    def __init__(self, factory_dict: dict[str, Callable[[], WordTokenizer]]):
        self._factory_dict = factory_dict
        self._tokenizers = {}

    def _get_tokenizer(self, language: str) -> WordTokenizer:
        if language not in self._tokenizers:
            if language not in self._factory_dict:
                raise ValueError(f"'{language}' tokenizer is not set.")
            tokenizer = self._factory_dict[language]()
            self._tokenizers[language] = tokenizer
        return self._tokenizers[language]

    @property
    def languages(self) -> list[str]:
        return list(self._factory_dict.keys())

    def word_tokenize(self, text: str, language: str) -> list[str]:
        return self._get_tokenizer(language).word_tokenize(text)

    def sent_tokenize(self, text: str, language: str) -> list[str]:
        return self._get_tokenizer(language).sent_tokenize(text)


WORD_TOKENIZER_FACTORY: dict[str, Callable[[], WordTokenizer]] = {
    Languages.english: lambda: NLTKTokenizer("english"),
    Languages.korean: lambda: SpaCyTokenizer("ko", {"nlp": {"tokenizer": {"@tokenizers": "spacy.Tokenizer.v1"}}}),
}

default_tokenizer = MultilingualTokenizer(WORD_TOKENIZER_FACTORY)
