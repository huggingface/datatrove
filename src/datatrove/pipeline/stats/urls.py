import json

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep
from datatrove.utils.stats import MetricStatsDict


class URLStats(PipelineStep):
    """Pipeline step to compute the statistics of URLs in a pipeline.
        Will add a "url_stats.json" file in the output folder with the statistics.

    Args:
        output_folder: the output folder to save the statistics
        url_field: the field to use as URL in the Document metadata (default: "url")
        input_folder: the input folder to read the statistics from (default: None). Used to merge statistics
        topk: the number of top URLs to keep (default: None - keep all)
        min_doc_count_to_save: the minimum number of documents per URL to save the URL (default: 1)
    """

    type = "📊 - STATS"
    name = "🌐 URLs"
    _requires_dependencies = ["tldextract"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        url_field: str = "url",
        input_folder: DataFolderLike = None,
        topk: int = None,
        min_doc_count_to_save: int = 1,
    ):
        super().__init__()
        self.url_field = url_field
        self.output_folder = get_datafolder(output_folder)
        self.input_folder = get_datafolder(input_folder) if input_folder else None
        self.topk = topk
        self.min_doc_count_to_save = min_doc_count_to_save

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        doc_counter = MetricStatsDict()
        tokens_counter = MetricStatsDict()
        total_docs = 0
        total_tokens = 0
        if self.input_folder:
            # reduce the map results
            assert world_size == 1, "world_size must be 1 when getting the input from an input_folder"
            for file in self.input_folder.list_files(glob_pattern="json"):
                with self.input_folder.open(file, "rt") as f:
                    file_data = json.load(f)
                    doc_counter += MetricStatsDict(init=file_data["doc_counter"])
                    tokens_counter += MetricStatsDict(init=file_data["tokens_counter"])
                    total_docs += file_data["total_docs"]
                    total_tokens += file_data["total_tokens"]
            if self.topk:
                doc_counter = doc_counter.topk(self.topk)
                tokens_counter = tokens_counter.topk(self.topk)
        else:
            from tldextract import tldextract

            # map and produce one output file per rank
            for doc in data:
                url = tldextract.extract(doc.metadata.get(self.url_field)).fqdn
                doc_counter[url] += 1
                total_docs += 1
                if token_count := doc.metadata.get("token_count", None):
                    tokens_counter[url] += token_count
                    total_tokens += token_count
                yield doc
        # save to disk
        if self.min_doc_count_to_save > 0:
            for url in list(doc_counter.keys()):
                if doc_counter[url].total < self.min_doc_count_to_save:
                    del doc_counter[url]
                    if url in tokens_counter:
                        del tokens_counter[url]
        with self.output_folder.open(
            f"{rank:05d}_url_stats.json" if not self.input_folder else "url_stats.json", "wt"
        ) as f:
            json.dump(
                {
                    "total_docs": total_docs,
                    "total_tokens": total_tokens,
                    "doc_counter": doc_counter.to_dict(),
                    "tokens_counter": tokens_counter.to_dict(),
                },
                f,
            )
