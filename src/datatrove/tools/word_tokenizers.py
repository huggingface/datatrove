from abc import ABC, abstractmethod

import spacy
from nltk.tokenize import word_tokenize

from datatrove.utils.typeshelper import Languages


class WordTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass


class NLTKTokenizer(WordTokenizer):
    def __init__(self, punkt_language: str):
        self.punkt_language = punkt_language

    def tokenize(self, text) -> list[str]:
        return word_tokenize(text, language=self.punkt_language)


class SpaCyTokenizer(WordTokenizer):
    def __init__(self, spacy_language: str, config=None):
        if config is None:
            self.tokenizer = spacy.blank(spacy_language)
        else:
            self.tokenizer = spacy.blank(spacy_language, config=config)

    def tokenize(self, text) -> list[str]:
        self.tokenizer.max_length = len(text) + 10
        return [
            token.text
            for token in self.tokenizer(text, disable=["parser", "tagger", "ner"])
            if len(token.text.strip()) > 0
        ]


WORD_TOKENIZERS: dict[str, WordTokenizer] = {
    Languages.english: NLTKTokenizer("english"),
    Languages.korean: SpaCyTokenizer("ko", {"nlp": {"tokenizer": {"@tokenizers": "spacy.Tokenizer.v1"}}}),
}


def get_word_tokenizer(language: str):
    if language in WORD_TOKENIZERS:
        return WORD_TOKENIZERS[language]
    else:
        return WORD_TOKENIZERS[Languages.english]
