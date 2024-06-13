from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.filters.gopher_quality_filter import STOP_WORDS
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


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
    stop_word_ratio: Ratio of stop words
    long_word_ratio_{chars}: Ratio of words longer than {chars} characters
    type_token_ratio: Type-Token Ratio (TTR)
    capitalized_word_ratio: Ratio of capitalized words
    uppercase_word_ratio: Ratio of uppercase words
    """

    name = "🈂️ Word stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        stop_words: list[str] = STOP_WORDS,
        short_word_max_chars_threshold: list[int] | None = None,
        long_word_max_chars_threshold: list[int] | None = None,
        language: str = Languages.english,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        histogram_round_digits: int = 3,
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
        self.language = language
        self.stop_words = stop_words

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        word_tokenizer = load_word_tokenizer(self.language)

        words = word_tokenizer.word_tokenize(doc.text)
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
            "type_token_ratio": len(set(words)) / len(words),
            "uppercase_word_ratio": sum([1 for word in words if word.isupper()]) / len(words),
            "capitalized_word_ratio": sum([1 for word in words if word.istitle()]) / len(words),
            "stop_word_ratio": sum([1 for word in words if word in self.stop_words]) / len(words),
        }
