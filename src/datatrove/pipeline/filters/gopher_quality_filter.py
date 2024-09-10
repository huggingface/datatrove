import numpy as np

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import PRECALCULATED_STATS, BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.logging import logger
from datatrove.utils.text import PUNCTUATION_SET
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


STOP_WORDS = ["the", "be", "to", "of", "and", "that", "have", "with"]


class GopherQualityFilter(BaseFilter):
    name = "🥇 Gopher Quality"

    def __init__(
        self,
        precalculated_stats: PRECALCULATED_STATS = PRECALCULATED_STATS.re_calculate_if_missing,
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
        language: str = Languages.english,
    ):
        """
        Filter to apply Gopher's quality heuristic rules.
        Reference: https://arxiv.org/pdf/2112.11446.pdf

        Args:
            min_doc_words:
            max_doc_words:
            min_avg_word_length:
            max_avg_word_length:
            max_symbol_word_ratio:
            max_bullet_lines_ratio:
            max_ellipsis_lines_ratio:
            max_non_alpha_words_ratio:
            min_stop_words:
            stop_words:
            exclusion_writer:
        """
        super().__init__(exclusion_writer)
        self.precalculated_stats = precalculated_stats
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
        self.tokenizer = load_word_tokenizer(language)

    def _filter_from_existing_stats(self, doc: Document) -> bool | tuple[bool, str]:
        if self.min_doc_words:
            if "n_non_symbol_words" not in doc.metadata["gopher"]:
                logger.warning(
                    f"Missing 'n_non_symbol_words' in doc metadata for {doc.id}."
                    "Ensure that the previous enrisher war run with `n_non_symbol_words` enabled."
                )
                return False, "missing_n_non_symbol_words"
            if (
                doc.metadata["gopher"]["n_non_symbol_words"] is not None
                and doc.metadata["gopher"]["n_non_symbol_words"] < self.min_doc_words
            ):
                return False, "gopher_short_doc"

        if self.max_doc_words:
            if "n_non_symbol_words" not in doc.metadata["gopher"]:
                logger.warning(
                    f"Missing 'n_non_symbol_words' in doc metadata for {doc.id}."
                    "Ensure that the previous enrisher war run with `n_non_symbol_words` enabled."
                )
                return False, "missing_n_non_symbol_words"
            if (
                doc.metadata["gopher"]["n_non_symbol_words"] is not None
                and doc.metadata["gopher"]["n_non_symbol_words"] > self.max_doc_words
            ):
                return False, "gopher_long_doc"

        if self.min_avg_word_length:
            if "avg_word_length" not in doc.metadata["gopher"]:
                logger.warning(
                    f"Missing 'avg_word_length' in doc metadata for {doc.id}."
                    "Ensure that the previous enrisher war run with `avg_word_length` enabled."
                )
                return False, "missing_avg_word_length"
            if (
                doc.metadata["gopher"]["avg_word_length"] is not None
                and doc.metadata["gopher"]["avg_word_length"] < self.min_avg_word_length
            ):
                return False, "gopher_below_avg_threshold"

        if self.max_avg_word_length:
            if "avg_word_length" not in doc.metadata["gopher"]:
                logger.warning(
                    f"Missing 'avg_word_length' in doc metadata for {doc.id}."
                    "Ensure that the previous enrisher war run with `avg_word_length` enabled."
                )
                return False, "missing_avg_word_length"
            if (
                doc.metadata["gopher"]["avg_word_length"] is not None
                and doc.metadata["gopher"]["avg_word_length"] > self.max_avg_word_length
            ):
                return False, "gopher_above_avg_threshold"

        if self.max_symbol_word_ratio:
            if "hash_to_word_ratio" not in doc.metadata["gopher"]:
                logger.warning(
                    f"Missing 'hash_to_word_ratio' in doc metadata for {doc.id}."
                    "Ensure that the previous enrisher war run with `hash_to_word_ratio` enabled."
                )
                return False, "missing_hash_to_word_ratio"
            if (
                doc.metadata["gopher"]["hash_to_word_ratio"] is not None
                and doc.metadata["gopher"]["hash_to_word_ratio"] > self.max_symbol_word_ratio
            ):
                return False, "gopher_too_many_hashes"

        if self.max_symbol_word_ratio:
            if "ellipsis_to_word_ratio" not in doc.metadata["gopher"]:
                logger.warning(
                    f"Missing 'ellipsis_to_word_ratio' in doc metadata for {doc.id}."
                    "Ensure that the previous enrisher war run with `ellipsis_to_word_ratio` enabled."
                )
                return False, "missing_ellipsis_to_word_ratio"
            if (
                doc.metadata["gopher"]["ellipsis_to_word_ratio"] is not None
                and doc.metadata["gopher"]["ellipsis_to_word_ratio"] > self.max_symbol_word_ratio
            ):
                return False, "gopher_too_many_ellipsis"

        if self.max_bullet_lines_ratio:
            if "bullet_lines_ratio" not in doc.metadata["gopher"]:
                logger.warning(
                    f"Missing 'bullet_lines_ratio' in doc metadata for {doc.id}."
                    "Ensure that the previous enrisher war run with `bullet_lines_ratio` enabled."
                )
                return False, "missing_bullet_lines_ratio"
            if (
                doc.metadata["gopher"]["bullet_lines_ratio"] is not None
                and doc.metadata["gopher"]["bullet_lines_ratio"] > self.max_bullet_lines_ratio
            ):
                return False, "gopher_too_many_bullets"

        if self.max_ellipsis_lines_ratio:
            if "end_ellipsis_ratio" not in doc.metadata["gopher"]:
                logger.warning(
                    f"Missing 'end_ellipsis_ratio' in doc metadata for {doc.id}."
                    "Ensure that the previous enrisher war run with `end_ellipsis_ratio` enabled."
                )
                return False, "missing_end_ellipsis_ratio"
            if (
                doc.metadata["gopher"]["end_ellipsis_ratio"] is not None
                and doc.metadata["gopher"]["end_ellipsis_ratio"] > self.max_ellipsis_lines_ratio
            ):
                return False, "gopher_too_many_end_ellipsis"

        if self.max_non_alpha_words_ratio:
            if "non_alpha_words_ratio" not in doc.metadata["gopher"]:
                logger.warning(
                    f"Missing 'non_alpha_words_ratio' in doc metadata for {doc.id}."
                    "Ensure that the previous enrisher war run with `non_alpha_words_ratio` enabled."
                )
                return False, "missing_non_alpha_words_ratio"
            if (
                doc.metadata["gopher"]["non_alpha_words_ratio"] is not None
                and doc.metadata["gopher"]["non_alpha_words_ratio"] < self.max_non_alpha_words_ratio
            ):
                return False, "gopher_below_alpha_threshold"

        if self.min_stop_words:
            if "stop_words_count" not in doc.metadata["gopher"]:
                logger.warning(
                    f"Missing 'stop_words_count' in doc metadata for {doc.id}."
                    "Ensure that the previous enrisher war run with `stop_words_count` enabled."
                )
                return False, "missing_stop_words_count"
            if (
                doc.metadata["gopher"]["stop_words_count"] is not None
                and doc.metadata["gopher"]["stop_words_count"] < self.min_stop_words
            ):
                return False, "gopher_enough_stop_words"

        return True

    def _filter_maybe_from_existing_stats(self, doc: Document) -> bool | tuple[bool, str]:
        """

        Args:
            doc: Applies the heuristics rules to decide if a document should be REMOVED


        Returns: False if sample.text does not pass any of the the heuristic tests

        """
        text = doc.text
        words = None
        n_words = None
        _force_recalc = False
        if self.precalculated_stats == PRECALCULATED_STATS.re_calculate:
            _force_recalc = True
            words = self.tokenizer.word_tokenize(text)
            n_words = len(words)

        # words < min_doc_words or words > max_doc_words
        if self.min_doc_words:
            if "n_non_symbol_words" not in doc.metadata["gopher"] or _force_recalc:
                if words is None:
                    words = self.tokenizer.word_tokenize(text)
                    n_words = len(words)
                n_non_symbol_words = len([w for w in words if any(ch not in PUNCTUATION_SET for ch in w)])
            else:
                n_non_symbol_words = doc.metadata["gopher"]["n_non_symbol_words"]
            if n_non_symbol_words < self.min_doc_words:
                return False, "gopher_short_doc"

        if self.max_doc_words:
            if "n_non_symbol_words" not in doc.metadata["gopher"] or _force_recalc:
                if words is None:
                    words = self.tokenizer.word_tokenize(text)
                    n_words = len(words)
                n_non_symbol_words = len([w for w in words if any(ch not in PUNCTUATION_SET for ch in w)])
            else:
                n_non_symbol_words = doc.metadata["gopher"]["n_non_symbol_words"]
            if n_non_symbol_words > self.max_doc_words:
                return False, "gopher_long_doc"

        # mean word length is outside the range of 3 to 10 characters
        non_symbol_words = None
        avg_n_words = None
        if self.min_avg_word_length:
            if "mean_word_length" not in doc.metadata["gopher"] or _force_recalc:
                if words is None:
                    words = self.tokenizer.word_tokenize(text)
                non_symbol_words = [w for w in words if any(ch not in PUNCTUATION_SET for ch in w)]
                avg_n_words = np.mean([len(w) for w in non_symbol_words])
            else:
                avg_n_words = doc.metadata["gopher"]["mean_word_length"]
            if avg_n_words < self.min_avg_word_length:
                return False, "gopher_below_avg_threshold"

        if self.max_avg_word_length:
            if "mean_word_length" not in doc.metadata["gopher"] or _force_recalc:
                if words is None:
                    words = self.tokenizer.word_tokenize(text)
                if non_symbol_words is None:
                    non_symbol_words = [w for w in words if any(ch not in PUNCTUATION_SET for ch in w)]
                if avg_n_words is None:
                    avg_n_words = np.mean([len(w) for w in non_symbol_words])
            else:
                avg_n_words = doc.metadata["gopher"]["mean_word_length"]
            if avg_n_words > self.max_avg_word_length:
                return False, "gopher_above_avg_threshold"

        # symbol-to-word ratio greater than 0.1 for either the hash symbol or the ellipsis
        if self.max_symbol_word_ratio:
            if "hash_to_word_ratio" not in doc.metadata["gopher"] or _force_recalc:
                hash_to_word_ratio = text.count("#") / n_words
            else:
                hash_to_word_ratio = doc.metadata["gopher"]["hash_to_word_ratio"]
            if hash_to_word_ratio > self.max_symbol_word_ratio:
                return False, "gopher_too_many_hashes"

        if self.max_symbol_word_ratio:
            if "ellipsis_to_word_ratio" not in doc.metadata["gopher"] or _force_recalc:
                ellipsis_to_word_ratio = (text.count("...") + text.count("…")) / n_words
            else:
                ellipsis_to_word_ratio = doc.metadata["gopher"]["ellipsis_to_word_ratio"]
            if ellipsis_to_word_ratio > self.max_symbol_word_ratio:
                return False, "gopher_too_many_ellipsis"

        # any document with more than 90 % of lines starting with a bullet point,
        # or more than 30 % ending with an ellipsis.
        if self.max_bullet_lines_ratio or self.max_ellipsis_lines_ratio:
            lines = text.splitlines()

        if self.max_bullet_lines_ratio:
            if "bullet_lines_ratio" not in doc.metadata["gopher"] or _force_recalc:
                bullet_lines_ratio = sum(
                    s.lstrip().startswith("•") or s.lstrip().startswith("-") for s in lines
                ) / len(lines)
            else:
                bullet_lines_ratio = doc.metadata["gopher"]["bullet_lines_ratio"]
            if bullet_lines_ratio > self.max_bullet_lines_ratio:
                return False, "gopher_too_many_bullets"

        if self.max_ellipsis_lines_ratio:
            if "end_ellipsis_ratio" not in doc.metadata["gopher"] or _force_recalc:
                end_ellipsis_ratio = sum(s.rstrip().endswith("...") or s.rstrip().endswith("…") for s in lines) / len(
                    lines
                )
            else:
                end_ellipsis_ratio = doc.metadata["gopher"]["end_ellipsis_ratio"]
            if end_ellipsis_ratio > self.max_ellipsis_lines_ratio:
                return False, "gopher_too_many_end_ellipsis"

        # ensure that 80 % of words in a document contain at least one alphabetic character
        if self.max_non_alpha_words_ratio:
            if "non_alpha_words_ratio" not in doc.metadata["gopher"] or _force_recalc:
                non_alpha_words_ratio = sum([any((c.isalpha() for c in w)) for w in words]) / n_words
            else:
                non_alpha_words_ratio = doc.metadata["gopher"]["non_alpha_words_ratio"]
            if non_alpha_words_ratio < self.max_non_alpha_words_ratio:
                return False, "gopher_below_alpha_threshold"

        # stop word filter
        if self.min_stop_words:
            if "stop_words_count" not in doc.metadata["gopher"] or _force_recalc:
                stop_words_count = sum(w in self.stop_words for w in words)
            else:
                stop_words_count = doc.metadata["gopher"]["stop_words_count"]
            if stop_words_count < self.min_stop_words:
                return False, "gopher_enough_stop_words"

        return True

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        if (
            self.precalculated_stats == PRECALCULATED_STATS.re_calculate
            or self.precalculated_stats == PRECALCULATED_STATS.re_calculate_if_missing
        ):
            return self._filter_maybe_from_existing_stats(doc)
        elif self.precalculated_stats == PRECALCULATED_STATS.re_use:
            if "gopher" not in doc.metadata:
                logger.warning(f"gopher not found in metadata for {doc.id}.")
                return False, "missing_gopher_field"
            return self._filter_from_existing_stats(doc)
        else:
            raise ValueError(f"Unknown precalculated_stats: {self.precalculated_stats}")
