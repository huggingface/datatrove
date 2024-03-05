from contextlib import nullcontext
from typing import Callable

from tqdm import tqdm

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers.base import BaseReader


class HuggingFaceDatasetReader(BaseReader):
    """Read data from HuggingFace datasets.
        Will read each row as a separate document.

    Args:
        dataset: the name of the dataset to load with datasets.load_dataset
        dataset_options: options to pass to the load_dataset function
        limit: limit the number of rows to read
        batch_size: the batch size to use
        progress: show progress bar
        adapter: function to adapt the data dict from the source to a Document.
            Take as input: data: dict, path: str, id_in_file: int | str
            Return: a dict with at least a "text" key
        text_key: key to use for the text in the default adapter (default: "text"). Ignored if you provide your own `adapter`
        id_key: key to use for the id in the default adapter (default: "id"). Ignored if you provide your own `adapter`
        default_metadata: default metadata to add to all documents
    """

    name = "🤗 HuggingFace"
    _requires_dependencies = ["datasets"]

    def __init__(
        self,
        dataset: str,
        dataset_options: dict | None = None,
        limit: int = -1,
        batch_size: int = 1000,
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
    ):
        super().__init__(limit, progress, adapter, text_key, id_key, default_metadata)
        self.dataset = dataset
        self.dataset_options = dataset_options
        self.batch_size = batch_size

    def get_document_from_dict(self, data: dict, source: str, id_in_file: int | str):
        document = super().get_document_from_dict(data, source, id_in_file)
        if document:
            document.metadata.setdefault("dataset", source)
        return document

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        from datasets import load_dataset  # type: ignore

        if data:
            yield from data
        # sadly sharding in this way with streaming is not supported by HF datasets yet, so no streaming
        ds = load_dataset(self.dataset, **self.dataset_options)
        shard = ds.shard(world_size, rank, contiguous=True)
        with tqdm(total=self.limit if self.limit != -1 else None) if self.progress else nullcontext() as pbar:
            li = 0
            for batch in shard.iter(self.batch_size):
                if self.limit != -1 and li >= self.limit:
                    break
                documents = []
                with self.track_time("batch"):
                    for line in (dict(zip(batch, t)) for t in zip(*batch.values())):
                        if self.limit != -1 and li >= self.limit:
                            break
                        document = self.get_document_from_dict(line, self.dataset, f"{rank:05d}/{li}")
                        if not document:
                            continue
                        documents.append(document)
                        self.update_doc_stats(document)
                        self.stat_update("documents")
                        li += 1
                        if self.progress:
                            pbar.update()
                yield from documents
