import json

from tldextract import tldextract

from datatrove.io import BaseInputDataFolder, BaseOutputDataFolder
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep
from datatrove.utils.stats import OnlineStatsDict


class URLStats(PipelineStep):
    type = "📊 - STATS"
    name = "🌐 URLs"

    def __init__(
        self,
        output_folder: BaseOutputDataFolder,
        url_field: str = "url",
        input_folder: BaseInputDataFolder = None,
        topk: int = None,
    ):
        super().__init__()
        self.url_field = url_field
        self.output_folder = output_folder
        self.input_folder = input_folder
        self.topk = topk

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        doc_counter = OnlineStatsDict()
        tokens_counter = OnlineStatsDict()
        if self.input_folder:
            # reduce the map results
            assert world_size == 1, "world_size must be 1 when getting the input from an input_folder"
            for file in self.input_folder.list_files(extension="json"):
                with file.open() as f:
                    file_data = json.load(f)
                    doc_counter += OnlineStatsDict(init=file_data["doc_counter"])
                    tokens_counter += OnlineStatsDict(init=file_data["tokens_counter"])
            if self.topk:
                doc_counter = doc_counter.topk(self.topk)
                tokens_counter = tokens_counter.topk(self.topk)
        else:
            # map and produce one output file per rank
            for doc in data:
                url_info = tldextract.extract(doc.metadata.get(self.url_field))
                url = ".".join([url_info.domain, url_info.subdomain])
                doc_counter[url] += 1
                if token_count := doc.metadata.get("token_count", None):
                    tokens_counter[url] += token_count
                yield doc
        # save to disk
        with self.output_folder.open(f"{rank:05d}_url_stats.json" if not self.input_folder else "url_stats.json") as f:
            json.dump({"doc_counter": doc_counter.to_dict(), "tokens_counter": tokens_counter.to_dict()}, f)
