from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.summary_stats.base import BaseStats
from datatrove.pipeline.stats.summary_stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from datatrove.utils.text import TextNormConfig, simplify_text


class WordsContaminationStats(BaseStats):
    """
    Words contamination stats of a document.

    Available stats:
    {words[0]}: Frequency of words contamination in the document.

    Args:
        words: The words to check for contamination.
    """

    name = "😷 Words contamination"
    _requires_dependencies = ["nltk"] + BaseStats._requires_dependencies

    def __init__(
        self,
        output_folder: DataFolderLike,
        words: list[str],
        norm_config: TextNormConfig = TextNormConfig(),
        histogram_round_digits: int = 3,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config=top_k_config)
        if len(words) == 0:
            raise ValueError("At least one word must be provided")

        self.norm_config = norm_config

        self.words = words

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        from nltk.tokenize import word_tokenize

        doc_words = word_tokenize(simplify_text(doc.text, self.norm_config))
        return {
            f"words_contamination_{self.words[0]}": sum([1 for word in doc_words if word in self.words])
            / len(doc_words)
        }
