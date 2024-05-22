import os
from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike, cached_asset_path_or_download
from datatrove.pipeline.stats.summary_stats.base import BaseStats
from datatrove.pipeline.stats.summary_stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from datatrove.utils.perplexity import KenlmModel

class CCNetPerplexityStats(BaseStats):
    """
    Summary stats of perplexity metrics:

    Available stats:
    ccnet_perplexity_{model_name}
    """
    name = "🤯 CCNet perplexity stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        model_name: str,
        language: str = "en",
        histogram_round_digits: int = 3,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)
        self.model = KenlmModel(model_dataset=model_name, language=language)
    
    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        return {f"ccnet_perplexity_{self.model.model_dataset}_{self.model.language}": self.model.get_perplexity(doc.text)}
