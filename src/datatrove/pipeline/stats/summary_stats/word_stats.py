from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.summary_stats.base import BaseStats
from datatrove.pipeline.stats.summary_stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig


def get_short_word_ratio(words: list[str], threshold: int) -> float:
    return sum([1 for word in words if len(word) <= threshold]) / len(words)


def get_long_word_ratio(words: list[str], threshold: int) -> float:
    return sum([1 for word in words if len(word) >= threshold]) / len(words)


class WordStats(BaseStats):
    """
    Word level stats of a document.

    Available stats:
    n_words: Number of words in the document
    avg_word_length: Average length of words in the document
    avg_words_per_line: Average number of words per line in the document
    short_word_ratio_{chars}: Ratio of words shorter than {chars} characters
    long_word_ratio_{chars}: Ratio of words longer than {chars} characters
    """

    name = "🈂️ Word stats"
    _requires_dependencies = ["nltk"] + BaseStats._requires_dependencies

    def __init__(
        self,
        output_folder: DataFolderLike,
        short_word_max_chars_threshold: list[int] | None = None,
        long_word_max_chars_threshold: list[int] | None = None,
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

        self.short_word_max_chars_threshold = short_word_max_chars_threshold or [3]
        self.long_word_max_chars_threshold = long_word_max_chars_threshold or [7]

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        from nltk.tokenize import word_tokenize

        words = word_tokenize(doc.text)
        lines = doc.text.splitlines()

        return {
            "n_words": len(words),
            "avg_word_length": sum([len(word) for word in words]) / len(words),
            "avg_words_per_line": len(words) / len(lines),
            **{
                f"short_word_ratio_{chars}": get_short_word_ratio(words, chars)
                for chars in self.short_word_max_chars_threshold
            },
            **{
                f"long_word_ratio_{chars}": get_long_word_ratio(words, chars)
                for chars in self.long_word_max_chars_threshold
            },
        }
