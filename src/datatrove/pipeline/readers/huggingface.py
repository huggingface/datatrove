from contextlib import nullcontext
from typing import Callable

from tqdm import tqdm

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers.base import BaseReader


class HuggingFaceReader(BaseReader):
    name = "🤗 HuggingFace"
    _requires_dependencies = ["datasets"]

    def __init__(
        self,
        dataset: str,
        dataset_options: dict | None = None,
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
    ):
        super().__init__(limit, progress, adapter, text_key, id_key, default_metadata)
        self.dataset = dataset
        self.dataset_options = dataset_options

    def get_document_from_dict(self, data: dict, source: str, id_in_file: int | str):
        document = super().get_document_from_dict(data, source, id_in_file)
        document.metadata.setdefault("dataset", source)
        return document

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        from datasets import load_dataset

        if data:
            yield from data
        # sadly sharding in this way with streaming is not supported by HF datasets yet, so no streaming
        ds = load_dataset(self.dataset, **self.dataset_options)
        shard = ds.shard(world_size, rank, contiguous=True)
        with tqdm(total=self.limit if self.limit != -1 else None) if self.progress else nullcontext() as pbar:
            for di, document in enumerate(shard):
                if self.limit != -1 and di >= self.limit:
                    break

                with self.track_time():
                    document = self.get_document_from_dict(document, self.dataset, f"{rank:05d}/{di}")
                    if not document:
                        continue
                self.update_doc_stats(document)
                self.stat_update("documents")
                yield document
                if self.progress:
                    pbar.update()
