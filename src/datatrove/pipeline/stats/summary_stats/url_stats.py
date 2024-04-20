import json

from datatrove.data import Document
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep
from datatrove.pipeline.stats.new_stats import GROUP, SummaryStats


class TokenStats(SummaryStats):
    """
    Token stats of a document.

    Available stats:
    tokens: Number of tokens in the document
    """

    type = "📊 - STATS"
    name = "🔗 Token counter"

    def __init__(
        self,
        output_folder: DataFolderLike,
        groups_to_compute: list[GROUP] = ["fqdn", "suffix", "summary", "histogram"],
    ) -> None:
        super().__init__(
            output_folder, groups_to_compute, round_digits=3
        )
    
    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        return {
            "token_count": doc.metadata.get("token_count", 0)
        }

