from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


def get_short_sentence_ratio(sentences: list[str], threshold: int) -> float:
    return sum([1 for sentence in sentences if len(sentence) <= threshold]) / len(sentences)


def get_long_sentence_ratio(sentences: list[str], threshold: int) -> float:
    return sum([1 for sentence in sentences if len(sentence) >= threshold]) / len(sentences)


class SentenceStats(BaseStats):
    """
    Sentence level stats of a document.

    Available stats:
    * n_sentences
    * avg_sentence_length:
    * short_sentence_ratio_{chars}:
    * long_sentence_ratio_{chars}:
    """

    name = "🈂️ Sentence stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        short_sentence_max_chars_threshold: list[int] | None = None,
        long_sentence_max_chars_threshold: list[int] | None = None,
        language: str = Languages.english,
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

        self.short_sentence_max_chars_threshold = short_sentence_max_chars_threshold or [20]
        self.long_sentence_max_chars_threshold = long_sentence_max_chars_threshold or [75]
        self.language = language

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        word_tokenizer = load_word_tokenizer(self.language)

        sentences = [s for s in word_tokenizer.sent_tokenize(doc.text) if s.strip()]

        return {
            "n_sentences": len(sentences),
            "avg_sentence_length": sum([len(s) for s in sentences]) / len(sentences),
            **{
                f"short_sentence_ratio_{chars}": get_short_sentence_ratio(sentences, chars)
                for chars in self.short_sentence_max_chars_threshold
            },
            **{
                f"long_sentence_ratio_{chars}": get_long_sentence_ratio(sentences, chars)
                for chars in self.long_sentence_max_chars_threshold
            },
        }
