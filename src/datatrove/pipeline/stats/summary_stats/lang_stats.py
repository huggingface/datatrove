from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.summary_stats.base import BaseStats
from datatrove.pipeline.stats.summary_stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from datatrove.utils.lid import FastTextModel


class LangStats(BaseStats):
    """
    Summary stats of language metrics:

    Available stats:
    fasttext_{language}
    """

    type = "📊 - STATS"
    name = "🎤 Language stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        language: str,
        histogram_round_digits: int = 3,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)
        self.fasttext = FastTextModel([language])
        self.language = language

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        return {f"fasttext_{self.language}": self.fasttext.predict(doc)[1][self.language]}
