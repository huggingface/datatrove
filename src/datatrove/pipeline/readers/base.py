from abc import abstractmethod
from contextlib import nullcontext
from typing import Callable

from loguru import logger
from tqdm import tqdm

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep


class BaseReader(PipelineStep):
    type = "📖 - READER"

    def __init__(
        self,
        data_folder: DataFolderLike,
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        content_key: str = "content",
        id_key: str = "data_id",
        default_metadata: dict = None,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.limit = limit
        self.progress = progress
        self.content_key = content_key
        self.id_key = id_key
        self.adapter = adapter if adapter else self._default_adapter
        self._empty_warning = False
        self.default_metadata = default_metadata

    def _default_adapter(self, data: dict, path: str, id_in_file: int):
        return {
            "content": data.pop(self.content_key, ""),
            "data_id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": data.pop("metadata", {}) | data,  # remaining data goes into metadata
        }

    def get_document_from_dict(self, data: dict, source_file: str, id_in_file: int):
        parsed_data = self.adapter(data, source_file, id_in_file)
        if not parsed_data.get("content", None):
            if not self._empty_warning:
                self._empty_warning = True
                logger.warning("Found document without content, skipping.")
            return None
        document = Document(**parsed_data)
        document.metadata.setdefault("file_path", self.data_folder.to_absolute_paths(source_file))
        if self.default_metadata:
            document.metadata = self.default_metadata | document.metadata
        return document

    @abstractmethod
    def read_file(self, filepath: str):
        raise NotImplementedError

    def read_files_shard(self, shard):
        li = 0
        with tqdm(total=self.limit if self.limit != -1 else None) if self.progress else nullcontext() as pbar:
            for filepath in shard:
                self.stat_update("input_files")
                logger.info(f"Reading input file {filepath}")
                di = 0
                for di, document in enumerate(self.read_file(filepath)):
                    if self.limit != -1 and li >= self.limit:
                        break
                    yield document
                    if self.progress:
                        pbar.update()
                    li += 1
                self.stat_update("documents", value=di, unit="input_file")
                if self.limit != -1 and li >= self.limit:
                    break

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data
        files_shard = self.data_folder.get_shard(rank, world_size)
        if len(files_shard) == 0:
            if rank == 0:
                raise RuntimeError(f"No files found on {self.data_folder.path}!")
            # otherwise just a warning
            logger.warning(f"No files found on {self.data_folder.path} for {rank=}")
        for doc in self.read_files_shard(files_shard):
            self.update_doc_stats(doc)
            yield doc
