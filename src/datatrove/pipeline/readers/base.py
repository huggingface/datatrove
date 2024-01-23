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
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
    ):
        super().__init__()
        self.limit = limit
        self.progress = progress
        self.text_key = text_key
        self.id_key = id_key
        self.adapter = adapter if adapter else self._default_adapter
        self._empty_warning = False
        self.default_metadata = default_metadata

    def _default_adapter(self, data: dict, path: str, id_in_file: int | str):
        return {
            "text": data.pop(self.text_key, ""),
            "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": data.pop("metadata", {}) | data,  # remaining data goes into metadata
        }

    def get_document_from_dict(self, data: dict, source_file: str, id_in_file: int | str):
        parsed_data = self.adapter(data, source_file, id_in_file)
        if not parsed_data.get("text", None):
            if not self._empty_warning:
                self._empty_warning = True
                logger.warning(
                    f"Found document without text, skipping. " f'Is your `text_key` ("{self.text_key}") correct?'
                )
            return None
        document = Document(**parsed_data)
        if self.default_metadata:
            document.metadata = self.default_metadata | document.metadata
        return document

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        raise NotImplementedError


class BaseDiskReader(BaseReader):
    type = "📖 - READER"

    def __init__(
        self,
        data_folder: DataFolderLike,
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
    ):
        super().__init__(limit, progress, adapter, text_key, id_key, default_metadata)
        self.data_folder = get_datafolder(data_folder)
        self.recursive = recursive
        self.glob_pattern = glob_pattern

    def get_document_from_dict(self, data: dict, source_file: str, id_in_file: int):
        document = super().get_document_from_dict(data, source_file, id_in_file)
        if document:
            document.metadata.setdefault("file_path", self.data_folder.resolve_paths(source_file))
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
        files_shard = self.data_folder.get_shard(
            rank, world_size, recursive=self.recursive, glob_pattern=self.glob_pattern
        )
        if len(files_shard) == 0:
            if rank == 0:
                raise RuntimeError(f"No files found on {self.data_folder.path}!")
            # otherwise just a warning
            logger.warning(f"No files found on {self.data_folder.path} for {rank=}")
        for doc in self.read_files_shard(files_shard):
            self.update_doc_stats(doc)
            yield doc
