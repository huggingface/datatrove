import string

import numpy as np

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


STOP_WORDS = ["the", "be", "to", "of", "and", "that", "have", "with"]


class GopherQualityFilter(BaseFilter):
    name = "🥇 Gopher Quality"
    requires_dependencies = ["nltk"]

    def __init__(
        self,
        min_doc_words: int | None = 50,
        max_doc_words: int | None = 100000,
        min_avg_word_length: int | None = 3,
        max_avg_word_length: int | None = 10,
        max_symbol_word_ratio: float | None = 0.1,
        max_bullet_lines_ratio: float | None = 0.9,
        max_ellipsis_lines_ratio: float | None = 0.3,
        max_non_alpha_words_ratio: float | None = 0.8,
        min_stop_words: int | None = 2,
        stop_words: list[str] | None = None,
        exclusion_writer: DiskWriter = None,
    ):
        """
        Filter to apply Gopher's quality heuristic rules.
        Reference: https://arxiv.org/pdf/2112.11446.pdf

        @param min_doc_words:
        @param max_doc_words:
        @param min_avg_word_length:
        @param max_avg_word_length:
        @param max_symbol_word_ratio:
        @param max_bullet_lines_ratio:
        @param max_ellipsis_lines_ratio:
        @param max_non_alpha_words_ratio:
        @param min_stop_words:
        @param stop_words:
        """
        super().__init__(exclusion_writer)
        self.min_doc_words = min_doc_words
        self.max_doc_words = max_doc_words
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_symbol_word_ratio = max_symbol_word_ratio
        self.max_bullet_lines_ratio = max_bullet_lines_ratio
        self.max_ellipsis_lines_ratio = max_ellipsis_lines_ratio
        self.max_non_alpha_words_ratio = max_non_alpha_words_ratio
        self.min_stop_words = min_stop_words
        self.stop_words = set(STOP_WORDS if stop_words is None else stop_words)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """
            Applies the heuristics rules to decide if a document should be REMOVED:
                -

        :param doc
        :return: False if sample.content does not pass any of the the heuristic tests
        """
        from nltk.tokenize import word_tokenize

        text = doc.content
        words = word_tokenize(text)  # TODO we should use language id filter

        # words < min_doc_words or words > max_doc_words
        n_words = len([w for w in words if w not in string.punctuation])
        if self.min_doc_words and n_words < self.min_doc_words:
            return False, "gopher_short_doc"
        if self.max_doc_words and n_words > self.max_doc_words:
            return False, "gopher_long_doc"

        # mean word length is outside the range of 3 to 10 characters
        avg_n_words = np.mean([len(w) for w in words if w not in string.punctuation])
        if self.min_avg_word_length and avg_n_words < self.min_avg_word_length:
            return False, "gopher_below_avg_threshold"
        if self.max_avg_word_length and avg_n_words > self.max_avg_word_length:
            return False, "gopher_above_avg_threshold"

        # symbol-to-word ratio greater than 0.1 for either the hash symbol or the ellipsis
        if self.max_symbol_word_ratio and text.count("#") / n_words > self.max_symbol_word_ratio:
            return False, "gopher_too_many_hashes"
        if self.max_symbol_word_ratio and (text.count("...") + text.count("…")) / n_words > self.max_symbol_word_ratio:
            return False, "gopher_too_many_ellipsis"

        # any document with more than 90 % of lines starting with a bullet point,
        # or more than 30 % ending with an ellipsis.
        lines = text.splitlines()
        if (
            self.max_bullet_lines_ratio
            and sum(s.lstrip().startswith("•") or s.lstrip().startswith("-") for s in lines) / len(lines)
            > self.max_bullet_lines_ratio
        ):
            return False, "gopher_too_many_bullets"
        if (
            self.max_ellipsis_lines_ratio
            and sum(s.rstrip().endswith("...") or s.rstrip().endswith("…") for s in lines) / len(lines)
            > self.max_ellipsis_lines_ratio
        ):
            return False, "gopher_too_many_end_ellipsis"

        # that 80 % of words in a document contain at least one alphabetic character
        if (
            self.max_non_alpha_words_ratio
            and sum([any((c.isalpha() for c in w)) for w in words]) / n_words < self.max_non_alpha_words_ratio
        ):
            return False, "gopher_below_alpha_threshold"

        # stop word filter
        if self.min_stop_words and sum(w in self.stop_words for w in words) < self.min_stop_words:
            return False, "gopher_enough_stop_words"

        return True
