import re
from collections import Counter

import numpy as np

from datatrove.data import Document
from datatrove.pipeline.enrichers.base_enricher import BaseEnricher
from datatrove.utils.text import PUNCTUATION_SET
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


STOP_WORDS = ["the", "be", "to", "of", "and", "that", "have", "with"]


def get_n_grams(words: list[str], n: int) -> list[str]:
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def find_duplicates(x: list[str]) -> tuple[int, int]:
    unique_x = set()
    duplicate_chars = 0
    duplicate_elements = 0
    for element in x:
        if element in unique_x:
            duplicate_chars += len(element)
            duplicate_elements += 1

        else:
            unique_x.add(element)
    return duplicate_elements, duplicate_chars


def find_top_duplicate(x: list[str]) -> int:
    counter = Counter()
    for element in x:
        counter[element] += 1
    top_n_gram = counter.most_common(1)[0]
    return len(top_n_gram[0]) * top_n_gram[1]


def find_all_duplicate(words: list[str], n: int) -> int:
    n_words = len(words)
    unique = set()
    repeated_chars, idx = 0, 0
    while idx < n_words - n + 1:
        n_gram = "".join(words[idx : idx + n])
        if n_gram in unique:
            repeated_chars += len(n_gram)
            idx += n
        else:
            unique.add(n_gram)
            idx += 1
    assert repeated_chars <= len("".join(words))
    return repeated_chars


class GopherQualityEnricher(BaseEnricher):
    name = "🥇 Gopher Quality Enricher"

    def __init__(
        self,
        n_non_symbol_words: bool = True,
        avg_word_length: bool = True,
        hash_to_word_ratio: bool = True,
        ellipsis_to_word_ratio: bool = True,
        bullet_lines_ratio: bool = True,
        end_ellipsis_ratio: bool = True,
        non_alpha_words_ratio: bool = True,
        stop_words_count: bool = True,
        dup_line_frac: bool = True,
        dup_para_frac: bool = True,
        dup_line_char_frac: bool = True,
        dup_para_char_frac: bool = True,
        top_ngram_ratio: list[int] = [2, 3, 4],
        dup_ngram_ratio: list[int] = [5, 6, 7, 8, 9, 10],
        stop_words: list[str] = STOP_WORDS,
        language: str = Languages.english,
        batch_size: int = 1,
    ):
        """
        Filter to apply Gopher's quality heuristic rules.
        Reference: https://arxiv.org/pdf/2112.11446.pdf

        Args:

        """
        super().__init__(batch_size)
        self.n_non_symbol_words = n_non_symbol_words
        self.avg_word_length = avg_word_length
        self.hash_to_word_ratio = hash_to_word_ratio
        self.ellipsis_to_word_ratio = ellipsis_to_word_ratio
        self.bullet_lines_ratio = bullet_lines_ratio
        self.end_ellipsis_ratio = end_ellipsis_ratio
        self.non_alpha_words_ratio = non_alpha_words_ratio
        self.stop_words_count = stop_words_count
        self.stop_words = set(stop_words)
        self.dup_line_frac = dup_line_frac
        self.dup_para_frac = dup_para_frac
        self.dup_line_char_frac = dup_line_char_frac
        self.dup_para_char_frac = dup_para_char_frac
        self.top_ngram_ratio = top_ngram_ratio
        self.dup_ngram_ratio = dup_ngram_ratio
        self.paragraph_exp = re.compile(r"\n{2,}")
        self._line_splitter = re.compile("\n+")
        self.tokenizer = load_word_tokenizer(language)

    def enrich(self, doc: Document) -> Document:
        text = doc.text
        gopher_metadata = {}

        # saves a bit of time if no metadata is needed
        if (
            self.n_non_symbol_words
            or self.avg_word_length
            or self.hash_to_word_ratio
            or self.ellipsis_to_word_ratio
            or self.non_alpha_words_ratio
            or self.stop_words_count
        ):
            words = self.tokenizer.word_tokenize(text)
            n_words = len(words)

        if self.n_non_symbol_words or self.avg_word_length:
            non_symbol_words = [w for w in words if any(ch not in PUNCTUATION_SET for ch in w)]

        if self.n_non_symbol_words:
            n_non_symbol_words = len(non_symbol_words)
            gopher_metadata["n_non_symbol_words"] = n_non_symbol_words

        if self.avg_word_length:
            avg_word_length = np.mean([len(w) for w in non_symbol_words])
            gopher_metadata["avg_word_length"] = avg_word_length

        if self.hash_to_word_ratio:
            hash_to_word_ratio = text.count("#") / n_words
            gopher_metadata["hash_to_word_ratio"] = hash_to_word_ratio

        if self.ellipsis_to_word_ratio:
            ellipsis_to_word_ratio = (text.count("...") + text.count("…")) / n_words
            gopher_metadata["ellipsis_to_word_ratio"] = ellipsis_to_word_ratio

        if self.bullet_lines_ratio or self.end_ellipsis_ratio:
            lines = text.splitlines()

        if self.bullet_lines_ratio:
            bullet_lines_ratio = sum(s.lstrip().startswith("•") or s.lstrip().startswith("-") for s in lines) / len(
                lines
            )
            gopher_metadata["bullet_lines_ratio"] = bullet_lines_ratio

        if self.end_ellipsis_ratio:
            end_ellipsis_ratio = sum(s.rstrip().endswith("...") or s.rstrip().endswith("…") for s in lines) / len(
                lines
            )
            gopher_metadata["end_ellipsis_ratio"] = end_ellipsis_ratio

        if self.non_alpha_words_ratio:
            non_alpha_words_ratio = sum([any((c.isalpha() for c in w)) for w in words]) / n_words
            gopher_metadata["non_alpha_words_ratio"] = non_alpha_words_ratio

        if self.stop_words_count:
            stop_words_count = sum(w.lower() in self.stop_words for w in words)
            gopher_metadata["stop_words_count"] = stop_words_count

        if self.dup_para_frac or self.dup_para_char_frac:
            paragraphs = self.paragraph_exp.split(text.strip())
            paragraphs_duplicates, para_char_duplicates = find_duplicates(paragraphs)

        if self.dup_para_frac:
            dup_para_frac = paragraphs_duplicates / len(paragraphs)
            gopher_metadata["dup_para_frac"] = dup_para_frac

        if self.dup_para_char_frac:
            dup_para_char_frac = para_char_duplicates / len(text)
            gopher_metadata["dup_para_char_frac"] = dup_para_char_frac

        if self.dup_line_frac or self.dup_line_char_frac:
            lines = self._line_splitter.split(text)
            line_duplicates, char_duplicates = find_duplicates(lines)

        if self.dup_line_frac:
            dup_line_frac = line_duplicates / len(lines)
            gopher_metadata["dup_line_frac"] = dup_line_frac

        if self.dup_line_char_frac:
            dup_line_char_frac = char_duplicates / len(text)
            gopher_metadata["dup_line_char_frac"] = dup_line_char_frac

        if self.top_ngram_ratio:
            for n in self.top_ngram_ratio:
                n_grams = get_n_grams(words, n)
                if not n_grams:
                    # We set it to -1 if there are no n-grams since
                    # if we are writing to parquet, we can't have None
                    # before having a float value
                    gopher_metadata[f"top_{n}_gram"] = -1.0
                    continue
                top_char_length = find_top_duplicate(n_grams)
                ngram_ratio = top_char_length / len(text)
                gopher_metadata[f"top_{n}_gram"] = ngram_ratio

        if self.dup_ngram_ratio:
            for n in self.dup_ngram_ratio:
                n_duplicates_char = find_all_duplicate(words, n)
                ngram_ratio = n_duplicates_char / len(text)
                gopher_metadata[f"duplicated_{n}_n_grams"] = ngram_ratio

        doc.metadata["gopher"] = gopher_metadata
        return doc
