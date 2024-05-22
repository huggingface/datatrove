import os
from typing import get_args

from datatrove.data import Document
from datatrove.io import DataFolderLike, cached_asset_path_or_download
from datatrove.pipeline.stats.summary_stats.base import BaseStats
from datatrove.pipeline.stats.summary_stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig

class PerplexityStats(BaseStats):
    """
    Summary stats of perplexity metrics:

    Available stats:
    kenlm_perplexity_{model_name}
    """

    type = "📊 - STATS"
    name = "🎤 Language stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        model_name: str,
        histogram_round_digits: int = 3,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)
        import kenlm
        self.model_name = model_name
        self._kenlm_model = None
    
    @property
    def kenlm_model(self):
        import kenlm
        if self._kenlm_model is None:
            if os.path.exists(self.model_name):
                self._kenlm_model = kenlm.Model(self.model_name)
            else:
                model_file = cached_asset_path_or_download(self.model_name)
                self._kenlm_model = kenlm.Model(model_file)
        return self._kenlm_model
    
    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        return {f"perplexity_{self.language}": self.fasttext.predict(doc)[1][self.language]}
