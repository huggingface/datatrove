from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.filters.gopher_repetition_filter import find_duplicates
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.text import TERMINAL_PUNCTUATION, split_into_words
from datatrove.utils.typeshelper import Languages


class FineWebQualityFilter(BaseFilter):
    name = "🍷 FineWeb Quality"

    def __init__(
        self,
        exclusion_writer: DiskWriter | None = None,
        line_punct_thr: float = 0.12,
        line_punct_exclude_zero: bool = False,
        stop_chars: tuple[str, ...] | None = None,
        short_line_thr: float = 0.67,
        short_line_length: int = 30,
        char_duplicates_ratio: float = 0.01,
        new_line_ratio: float = 0.3,
        language: str = Languages.english,
    ):
        super().__init__(exclusion_writer)
        self.line_punct_thr = line_punct_thr
        self.line_punct_exclude_zero = line_punct_exclude_zero
        self.stop_chars = stop_chars if stop_chars is not None else tuple(TERMINAL_PUNCTUATION)
        self.short_line_threshold = short_line_thr
        self.short_line_length = short_line_length
        self.char_duplicates_ratio = char_duplicates_ratio
        self.new_line_ratio = new_line_ratio
        self.language = language

    def filter(self, doc) -> bool | tuple[bool, str]:
        lines = doc.text.split("\n")
        lines = [line for line in lines if line.strip() != ""]
        if len(lines) == 0:
            return False, "empty"
        ratio = sum(1 for line in lines if line.endswith(self.stop_chars)) / len(lines)
        if ratio < self.line_punct_thr and not (ratio == 0 and self.line_punct_exclude_zero):
            return False, "line_punct_ratio"

        ratio = sum(1 for line in lines if len(line) <= self.short_line_length) / len(lines)
        if ratio > self.short_line_threshold:
            return False, "short_line_ratio"

        ratio = find_duplicates(lines)[1] / len(doc.text.replace("\n", ""))

        if ratio > self.char_duplicates_ratio:
            return False, "char_dup_ratio"

        words = split_into_words(doc.text, self.language)
        new_line = doc.text.count("\n")
        if new_line / len(words) > self.new_line_ratio:
            return False, "list_ratio"

        return True
