from datatrove.pipeline.filters.base_filter import BaseFilter

class FineWebQualityFilter(BaseFilter):
    name = "fineweb quality filter"
    _requires_dependencies = ["nltk"]

    def __init__(
            self,
            exclusion_writer,
            line_punct_thr: float = 0.12,
            line_punct_exclude_zero=False,
            short_line_thr: float = 0.67,
            short_line_length: int = 30,
            char_duplicates_ratio: float = 0.01
        ):
        super().__init__(exclusion_writer)
        self.line_punct_thr = line_punct_thr
        self.line_punct_exclude_zero = line_punct_exclude_zero
        self.short_line_threshold = short_line_thr
        self.short_line_length = short_line_length
        self.char_duplicates_ratio = char_duplicates_ratio

    def filter(self, doc) -> bool | tuple[bool, str]:
        from datatrove.pipeline.filters.gopher_repetition_filter import find_duplicates

        def remove_empty_lines(lines: list[str]):
            return [l for l in lines if l.strip() != ""]

        stop_chars = (".", "'", '"', "!", "?")

        lines = doc.text.split("\n")
        ratio = sum(1 for line in lines if line.endswith(stop_chars)) / len(lines)
        if ratio <= self.line_punct_thr and not (ratio == 0 and self.line_punct_exclude_zero):
            return False, "line_punct_ratio"

        ratio = sum(1 for line in lines if len(line) <= self.short_line_length) / len(
            lines
        )
        if ratio >= self.short_line_threshold:
            return False, "short_line_ratio"

        ratio = find_duplicates(remove_empty_lines(lines))[1] / len(doc.text.replace("\n", ""))

        if ratio >= self.char_duplicates_ratio:
            return False, "char_dup_ratio"

        return True