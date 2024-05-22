from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.filters.gopher_quality_filter import STOP_WORDS
from datatrove.pipeline.stats.summary_stats.base import BaseStats
from datatrove.pipeline.stats.summary_stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from datatrove.utils.tokenization import load_tokenizer


class DocStats(BaseStats):
    """
    Summary stats of document level metrics:

    Available stats:
    length: Length of the document
    white_space_ratio: Ratio of whitespace characters
    non_alpha_digit_ratio: Ratio of non-alphabetic and non-digit characters
    digit_ratio: Ratio of digits
    stop_word_ratio: Ratio of stop words
    """

    type = "📊 - STATS"
    name = "📜 Doc stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        stop_words: list[str] = STOP_WORDS,
        histogram_round_digits: int = 3,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)
        self.stop_words = stop_words

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        from nltk.tokenize import word_tokenize
        words = word_tokenize(doc.text)
        return {
            "length": len(doc.text),
            "white_space_ratio": sum([1 for c in doc.text if c.isspace()]) / len(doc.text),
            "non_alpha_digit_ratio": sum([1 for c in doc.text if not c.isalpha() and not c.isdigit()]) / len(doc.text),
            "digit_ratio": sum([1 for c in doc.text if c.isdigit()]) / len(doc.text),
            "stop_word_ratio": sum([1 for word in words if word in self.stop_words]) / len(words),
        }
