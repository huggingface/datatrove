import re
from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from datatrove.utils.text import PUNCTUATION


ELIPSIS = ["...", "…"]


class DocStats(BaseStats):
    """
    Summary stats of document level metrics:

    Available stats:
    length: Length of the document
    white_space_ratio: Ratio of whitespace characters
    non_alpha_digit_ratio: Ratio of non-alphabetic and non-digit characters
    digit_ratio: Ratio of digits
    uppercase_ratio: Ratio of uppercase letters
    elipsis_ratio: Ratio of elipsis characters
    punctuation_ratio: Punctuation ratio
    """

    name = "📜 Doc stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        histogram_round_digits: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)
        self.elipsis_regex = re.compile("|".join([f"(?:{re.escape(elipsis)})" for elipsis in ELIPSIS]))
        self.punc_regex = re.compile("|".join([f"(?:{re.escape(punc)})" for punc in PUNCTUATION]))

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        return {
            "length": len(doc.text),
            "white_space_ratio": sum([1 for c in doc.text if c.isspace()]) / len(doc.text),
            "non_alpha_digit_ratio": sum([1 for c in doc.text if not c.isalpha() and not c.isdigit()]) / len(doc.text),
            "digit_ratio": sum([1 for c in doc.text if c.isdigit()]) / len(doc.text),
            "uppercase_ratio": sum([1 for c in doc.text if c.isupper()]) / len(doc.text),
            "elipsis_ratio": sum(len(elipsis) for elipsis in self.elipsis_regex.findall(doc.text)) / len(doc.text),
            "punctuation_ratio": sum(len(punc) for punc in self.punc_regex.findall(doc.text)) / len(doc.text),
        }
