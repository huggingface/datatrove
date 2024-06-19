from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from datatrove.utils.lid import FT176LID


class LangStats(BaseStats):
    """
    Summary stats of language metrics:

    Available stats:
    fasttext_{language}
    """

    name = "🎤 Language stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        language: str,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        histogram_round_digits: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)
        self.fasttext = FT176LID([language])
        self.language = language

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        language_score = 0
        if doc.metadata.get("language") == self.language and "language_score" in doc.metadata:
            language_score = doc.metadata["language_score"]
        else:
            language_score = self.fasttext.predict(doc)[1][self.language]
        return {f"fasttext_{self.language}": language_score}
