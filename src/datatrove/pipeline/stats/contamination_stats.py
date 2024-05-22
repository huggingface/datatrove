from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from datatrove.utils.text import TextNormConfig, simplify_text
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


class WordsContaminationStats(BaseStats):
    """
    Words contamination stats of a document.

    Available stats:
    word_contamination_{words[0]}: Frequency of words contamination in the document.

    Args:
        words: The words to check for contamination.
    """

    name = "😷 Words contamination"

    def __init__(
        self,
        output_folder: DataFolderLike,
        words: list[str],
        norm_config: TextNormConfig = TextNormConfig(),
        language: str = Languages.english,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        histogram_round_digits: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config=top_k_config)
        if len(words) == 0:
            raise ValueError("At least one word must be provided")

        self.norm_config = norm_config
        self.language = language
        self.words = words

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        word_tokenizer = load_word_tokenizer(self.language)

        doc_words = word_tokenizer.word_tokenize(simplify_text(doc.text, self.norm_config))
        return {
            f"words_contamination_{self.words[0]}": sum([1 for word in doc_words if word in self.words])
            / len(doc_words)
        }
