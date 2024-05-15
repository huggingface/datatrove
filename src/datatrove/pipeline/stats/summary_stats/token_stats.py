from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.summary_stats.base import BaseStats
from datatrove.pipeline.stats.summary_stats.config import GROUP
from datatrove.utils.tokenization import PipelineStepWithTokenizer


class TokenStats(BaseStats, PipelineStepWithTokenizer):
    """
    Token stats of a document.

    Available stats:
    token_count: Number of tokens in the document
    unique_token_count: Number of unique tokens in the document
    """

    name = "🔗 Token counter"

    _requires_dependencies = ["tokenizers"] + BaseStats._requires_dependencies

    def __init__(
        self,
        output_folder: DataFolderLike,
        tokenizer_name_or_path: str = "gpt2",
        groups_to_compute: list[GROUP] = ["fqdn", "suffix", "summary", "histogram"],
        histogram_rounding: int = 3,
    ) -> None:
        BaseStats.__init__(self, output_folder, groups_to_compute, histogram_rounding)
        PipelineStepWithTokenizer.__init__(self)
        self.tokenizer_name_or_path = tokenizer_name_or_path

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        tokens = doc.metadata.get("token_count", None)
        if tokens is None:
            tokens = self.tokenizer.encode(doc.text).tokens

        return {
            "token_count": len(tokens),
            "unique_token_count": len(set(tokens)),
        }
