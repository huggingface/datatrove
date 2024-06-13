from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.filters.gopher_repetition_filter import find_duplicates
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig


def get_short_paragraph_ratio(paragraphs: list[str], threshold: int) -> float:
    return sum([1 for paragraph in paragraphs if len(paragraph) <= threshold]) / len(paragraphs)


def get_long_paragraph_ratio(paragraphs: list[str], threshold: int) -> float:
    return sum([1 for paragraph in paragraphs if len(paragraph) >= threshold]) / len(paragraphs)


class ParagraphStats(BaseStats):
    """
    Summary stats of paragraphs in a document.

    Available stats:
    n_paragraphs
    avg_paragraph_length
    short_paragraph_ratio_{chars}
    long_paragraph_ratio_{chars}
    """

    type = "📊 - STATS"
    name = "📄 Paragraph stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        short_paragraph_max_chars_threshold: list[int] | None = None,
        long_paragraph_max_chars_threshold: list[int] | None = None,
        ignore_empty_paragraphs: bool = False,
        histogram_round_digits: int = 3,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(
            output_folder,
            groups_to_compute,
            histogram_round_digits,
            top_k_config,
        )

        self.ignore_empty_paragraphs = ignore_empty_paragraphs
        self.short_paragraph_max_chars_threshold = short_paragraph_max_chars_threshold or [100]
        self.long_paragraph_max_chars_threshold = long_paragraph_max_chars_threshold or [1000]

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        paragraphs = [p for p in doc.text.split("\n\n") if p.strip()]
        # Don't ignore empty paragraphs for count
        n_paragraphs = len(paragraphs)

        paragraphs = [p for p in paragraphs if p.strip()] if self.ignore_empty_paragraphs else paragraphs
        paragraph_dups, paragraph_char_dups = find_duplicates(paragraphs)

        return {
            "n_paragraphs": n_paragraphs,
            "avg_paragraph_length": sum([len(p) for p in paragraphs]) / n_paragraphs,
            **{
                f"short_paragraph_ratio_{chars}": get_short_paragraph_ratio(paragraphs, chars)
                for chars in self.short_paragraph_max_chars_threshold
            },
            **{
                f"long_paragraph_ratio_{chars}": get_long_paragraph_ratio(paragraphs, chars)
                for chars in self.long_paragraph_max_chars_threshold
            },
            "paragraph_duplicates": paragraph_dups / n_paragraphs,
            "paragraph_char_duplicates": paragraph_char_dups / sum(len(p) for p in paragraphs),
        }
