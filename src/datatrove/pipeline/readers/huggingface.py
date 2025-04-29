import copy
from typing import Callable

from loguru import logger
from tqdm import tqdm

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers.base import BaseReader


class HuggingFaceDatasetReader(BaseReader):
    """Read data from HuggingFace datasets.
        Will read each row as a separate document.

    Args:
        dataset: the name of the dataset to load with datasets.load_dataset
        dataset_options: options to pass to the load_dataset function
        streaming: whether to stream the dataset
        limit: limit the number of rows to read
        skip: skip the first n rows
        batch_size: the batch size to use
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Take as input: data: dict, path: str, id_in_file: int | str
            Return: a dict with at least a "text" key
        text_key: key to use for the text in the default adapter (default: "text"). Ignored if you provide your own `adapter`
        id_key: key to use for the id in the default adapter (default: "id"). Ignored if you provide your own `adapter`
        default_metadata: default metadata to add to all documents
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use
            with dedup blocks
    """

    name = "🤗 HuggingFace"
    _requires_dependencies = ["datasets"]

    def __init__(
        self,
        dataset: str,
        dataset_options: dict | None = None,
        streaming: bool = False,
        limit: int = -1,
        skip: int = 0,
        batch_size: int = 1000,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        shuffle_files: bool = False,
    ):
        super().__init__(limit, skip, adapter, text_key, id_key, default_metadata)
        self.dataset = dataset
        self.dataset_options = dataset_options or {}
        self.batch_size = batch_size
        self.doc_progress = doc_progress
        self.streaming = streaming
        self.shuffle_files = shuffle_files

    def get_document_from_dict(self, data: dict, source_file: str, id_in_file: int | str):
        document = super().get_document_from_dict(data, source_file, id_in_file)
        if document:
            document.metadata.setdefault("dataset", source_file)
        return document

    def _get_dataset_shard(self, dst, rank: int, world_size: int):
        from datasets import Dataset, IterableDataset
        from datasets.distributed import split_dataset_by_node

        if isinstance(dst, Dataset):
            return dst.shard(world_size, rank, contiguous=False)
        elif isinstance(dst, IterableDataset) and dst.n_shards > 1:
            # In case we have more than 1 shard (file), we shard
            # on shards/file level.
            if rank >= dst.n_shards:
                logger.warning(
                    f"Requested shard {rank} of a streaming dataset, but it only has {dst.n_shards} shards."
                )
                return None
            ex_iterable = dst._ex_iterable.shard_data_sources(index=rank, num_shards=world_size, contiguous=False)
            return IterableDataset(
                ex_iterable=ex_iterable,
                info=dst._info.copy(),
                split=dst._split,
                formatting=dst._formatting,
                shuffling=copy.deepcopy(dst._shuffling),
                distributed=copy.deepcopy(dst._distributed),
                token_per_repo_id=dst._token_per_repo_id,
            )
        else:
            # If we have just a single shard/file, we shard inter-file
            return split_dataset_by_node(dataset=dst, rank=rank, world_size=world_size)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        from datasets import load_dataset  # type: ignore

        if data:
            yield from data
        ds = load_dataset(self.dataset, **self.dataset_options, streaming=self.streaming)

        if self.shuffle_files:
            if not self.streaming:
                ds = ds.shuffle(seed=42)
            else:
                ds = ds.shuffle(seed=42, buffer_size=1000)

        # In case the dataset is (Iterable)?DatasetDict, raise informative error
        if isinstance(ds, dict):
            raise ValueError(
                f"You forgot to specify the split of the dataset. Update your dataset_options to include 'split'. Available splits: {list(ds.keys())}"
            )

        shard = self._get_dataset_shard(ds, rank, world_size)
        if not shard:
            return
        with tqdm(total=self.limit if self.limit != -1 else None, disable=not self.doc_progress) as pbar:
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
                        pbar.update()
                yield from documents
