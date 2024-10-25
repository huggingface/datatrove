from typing import Tuple

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import PRECALCULATED_STATS, BaseFilter
from datatrove.pipeline.filters.gopher_repetition_filter import find_duplicates
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


STOP_CHARS = (".", "'", '"', "!", "?")


class FineWebQualityFilter(BaseFilter):
    name = "🍷 FineWeb Quality"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
        precalculated_stats: PRECALCULATED_STATS = PRECALCULATED_STATS.re_calculate_if_missing,
        line_punct_thr: float = 0.12,
        line_punct_exclude_zero: bool = False,
        short_line_thr: float = 0.67,
        short_line_length: int = 30,
        char_duplicates_ratio: float = 0.01,
        new_line_ratio: float = 0.3,
        language: str = Languages.english,
        stop_chars: Tuple[str] = STOP_CHARS,
    ):
        super().__init__(exclusion_writer)
        self.precalculated_stats = precalculated_stats
        self.line_punct_thr = line_punct_thr
        self.line_punct_exclude_zero = line_punct_exclude_zero
        self.short_line_threshold = short_line_thr
        self.short_line_length = short_line_length
        self.char_duplicates_ratio = char_duplicates_ratio
        self.new_line_ratio = new_line_ratio
        self.tokenizer = load_word_tokenizer(language)
        self.stop_chars = stop_chars

    def _filter_from_existing_stats(self, doc: Document) -> bool | tuple[bool, str]:
        if self.line_punct_thr >= 0:
            line_punct_ratio = doc.metadata.get("fineweb", {}).get("line_punct_ratio")
            if line_punct_ratio is None:
                logger.warning(
                    f"Missing 'line_punct_ratio' in doc metadata for {doc.id}"
                    "Ensure that the previous enricher war run with `line_punct_ratio` enabled."
                )
                return False, "missing_line_punct_ratio"
            if line_punct_ratio <= self.line_punct_thr and not (
                line_punct_ratio == 0 and self.line_punct_exclude_zero
            ):
                return False, "line_punct_ratio"

        if self.short_line_threshold >= 0:
            line_length = doc.metadata.get("fineweb", {}).get("line_length")
            if line_length is None:
                logger.warning(
                    f"Missing 'line_length' in doc metadata for {doc.id}"
                    "Ensure that the previous enricher war run with `line_length` enabled."
                )
                return False, "missing_line_length"
            if (
                sum(1 for line in line_length if line <= self.short_line_length) / len(line_length)
                >= self.short_line_threshold
            ):
                return False, "short_line_ratio"

        if self.char_duplicates_ratio >= 0:
            char_dup_ratio = doc.metadata.get("fineweb", {}).get("char_duplicates_ratio")
            if char_dup_ratio is None:
                logger.warning(
                    f"Missing 'char_duplicates_ratio' in doc metadata for {doc.id}"
                    "Ensure that the previous enricher war run with `char_duplicates_ratio` enabled."
                )
                return False, "missing_char_duplicates_ratio"
            if char_dup_ratio >= self.char_duplicates_ratio:
                return False, "char_dup_ratio"

        if self.new_line_ratio >= 0:
            new_line_ratio = doc.metadata.get("fineweb", {}).get("new_line_ratio")
            if new_line_ratio is None:
                logger.warning(
                    f"Missing 'new_line_ratio' in doc metadata for {doc.id}"
                    "Ensure that the previous enricher war run with `new_line_ratio` enabled."
                )
                return False, "missing_new_line_ratio"
            if new_line_ratio >= self.new_line_ratio:
                return False, "new_line_ratio"

        return True

    def _filter_maybe_from_existing_stats(self, doc: Document) -> bool | tuple[bool, str]:
        lines = None

        _force_recalc = False
        if self.precalculated_stats == PRECALCULATED_STATS.re_calculate:
            _force_recalc = True
            lines = doc.text.split("\n")

        if self.line_punct_thr >= 0:
            if "line_punct_ratio" not in doc.metadata or _force_recalc:
                if lines is None:
                    lines = doc.text.split("\n")
                line_punct_ratio = sum(1 for line in lines if line.endswith(self.stop_chars)) / len(lines)
            else:
                line_punct_ratio = doc.metadata["line_punct_ratio"]

            if line_punct_ratio <= self.line_punct_thr and not (
                line_punct_ratio == 0 and self.line_punct_exclude_zero
            ):
                return False, "line_punct_ratio"

        if self.short_line_threshold >= 0:
            if "line_length" not in doc.metadata or _force_recalc:
                if lines is None:
                    lines = doc.text.split("\n")
                line_length = [len(line) for line in lines]
            else:
                line_length = doc.metadata["line_length"]

            if (
                sum(1 for line in line_length if line <= self.short_line_length) / len(line_length)
                >= self.short_line_threshold
            ):
                return False, "short_line_ratio"

        if self.char_duplicates_ratio >= 0:
            if "char_duplicates_ratio" not in doc.metadata or _force_recalc:
                non_empty_lines = [line for line in lines if line.strip() != ""]
                char_dup_ratio = find_duplicates(non_empty_lines)[1] / len(doc.text.replace("\n", ""))
            else:
                char_dup_ratio = doc.metadata["char_duplicates_ratio"]
            if char_dup_ratio >= self.char_duplicates_ratio:
                return False, "char_dup_ratio"

        if self.new_line_ratio >= 0:
            if "new_line_ratio" not in doc.metadata or _force_recalc:
                words = self.tokenizer.word_tokenize(doc.text)
                new_line = doc.text.count("\n")
                new_line_ratio = new_line / len(words)
            else:
                new_line_ratio = doc.metadata["new_line_ratio"]
            if new_line_ratio >= self.new_line_ratio:
                return False, "new_line_ratio"

        return True

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        if (
            self.precalculated_stats == PRECALCULATED_STATS.re_calculate
            or self.precalculated_stats == PRECALCULATED_STATS.re_calculate_if_missing
        ):
            return self._filter_maybe_from_existing_stats(doc)
        elif self.precalculated_stats == PRECALCULATED_STATS.re_use:
            if "fineweb" not in doc.metadata:
                logger.warning("fineweb not found in metadata.")
                return False, "missing_fineweb_field"
            return self._filter_from_existing_stats(doc)
        else:
            raise ValueError(f"Unknown precalculated_stats: {self.precalculated_stats}")
