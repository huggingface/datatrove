from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from datatrove.utils.tokenization import PipelineStepWithTokenizer


class TokenStats(BaseStats, PipelineStepWithTokenizer):
    """
    Token stats of a document.

    Available metrics:
    token_count: Number of tokens in the document
    """

    name = "🔗 Token counter"

    _requires_dependencies = ["tokenizers"] + BaseStats._requires_dependencies

    def __init__(
        self,
        output_folder: DataFolderLike,
        tokenizer_name_or_path: str = "gpt2",
        groups_to_compute: list[GROUP] = ["fqdn", "suffix", "summary", "histogram"],
        histogram_rounding: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        BaseStats.__init__(self, output_folder, groups_to_compute, histogram_rounding, top_k_config)
        PipelineStepWithTokenizer.__init__(self, tokenizer_name_or_path)

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        tokens_count = doc.metadata.get("token_count", None)
        if tokens_count is None:
            tokens_count = len(self.tokenizer.encode(doc.text).tokens)

        return {
            "token_count": tokens_count,
        }
