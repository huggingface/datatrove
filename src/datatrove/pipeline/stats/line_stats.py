from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.filters.c4_filters import END_PUNCTUATION
from datatrove.pipeline.filters.gopher_repetition_filter import find_duplicates
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig


def get_max_chars_per_line_ratio(lines, chars: int) -> float:
    return sum([1 for line in lines if len(line) <= chars]) / len(lines)


def get_min_chars_per_line_ratio(lines, chars: int) -> float:
    return sum([1 for line in lines if len(line) >= chars]) / len(lines)


def is_bullet_line(line: str):
    if len(line.strip()) == 0:
        return False
    return line.strip()[0] in "-*•"


class LineStats(BaseStats):
    """
    Summary stats of line level metrics.

    Available stats:
    n_lines: Number of lines per doc
    avg_line_length: Average length of line per doc
    long_line_ratio_chars_{chars}: Ratio of lines with more than k chars
    short_line_ratio_chars_{chars}: Ratio of lines with more than k chars
    bullet_point_lines_ratio: Ratio of bullet points
    line_duplicates: Ratio of lines that are duplicates
    line_char_duplicates: Ratio of chars in duplicated lines

    Args:
        max_k_chars_per_line_tresholds: List of max chars per line to compute stats for. If None, default to [10, 30]
        min_k_chars_per_line_thresholds: List of min chars per line to compute stats for. If None, default to [2000, 10000]
    """

    name = "🎼 Line stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        max_k_chars_per_line_tresholds: list[int] | None = None,
        min_k_chars_per_line_thresholds: list[int] | None = None,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        ignore_empty_lines: bool = False,
        histogram_round_digits: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)
        self.short_max_chars = (
            max_k_chars_per_line_tresholds if max_k_chars_per_line_tresholds is not None else [10, 30]
        )
        self.long_max_chars = (
            min_k_chars_per_line_thresholds if min_k_chars_per_line_thresholds is not None else [2000, 10000]
        )
        self.ignore_empty_lines = ignore_empty_lines

    def extract_stats(self, doc: Document):
        lines: list[str] = doc.metadata.get("lines") or doc.text.split("\n")
        # Don't ignore empty lines for count
        n_lines = len(lines)

        lines = [line for line in lines if len(line.strip()) > 0] if self.ignore_empty_lines else lines
        line_dups, char_dups = find_duplicates(lines)
        return {
            "n_lines": n_lines,
            "avg_line_length": (sum([len(line) for line in lines]) / len(lines)),
            **{
                f"short_line_ratio_chars_{chars}": get_max_chars_per_line_ratio(lines, chars)
                for chars in self.short_max_chars
            },
            **{
                f"long_line_ratio_chars_{chars}": get_min_chars_per_line_ratio(lines, chars)
                for chars in self.long_max_chars
            },
            "lines_ending_with_terminal_mark_ratio": sum(1 for line in lines if line.endswith(END_PUNCTUATION))
            / len(lines),
            "bullet_point_lines_ratio": sum(1 for line in lines if is_bullet_line(line)) / len(lines),
            "line_duplicates": line_dups / len(lines),
            "line_char_duplicates": char_dups / sum(len(line) for line in lines),
        }
