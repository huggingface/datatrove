import random
from abc import abstractmethod
from types import MethodType
from typing import Callable

from tqdm import tqdm

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger


class BaseReader(PipelineStep):
    """Base module for Readers. Readers read data from a source and create documents.
        Reader are the first step in a pipeline usually.

    Args:
        limit: limit the number of documents to read. Useful for debugging
        adapter: function to adapt the data dict from the source to a Document.
            Takes as input: (self, data: dict, path: str, id_in_file: int | str)
                self allows access to self.text_key and self.id_key
            Returns: a dict with at least a "text" key
        text_key: key to use for the text in the default adapter (default: "text").
        id_key: key to use for the id in the default adapter (default: "id").
        default_metadata: default metadata to add to all documents
    """

    type = "📖 - READER"

    def __init__(
        self,
        limit: int = -1,
        skip: int = 0,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
    ):
        super().__init__()
        self.limit = limit
        self.skip = skip
        self.text_key = text_key
        self.id_key = id_key
        self.adapter = MethodType(adapter, self) if adapter else self._default_adapter
        self._empty_warning = False
        self.default_metadata = default_metadata

    def _default_adapter(self, data: dict, path: str, id_in_file: int | str):
        """
        The default data adapter to adapt input data into the datatrove Document format

        Args:
            data: a dictionary with the "raw" representation of the data
            path: file path or source for this sample
            id_in_file: its id in this particular file or source

        Returns: a dictionary with text, id, media and metadata fields

        """
        return {
            "text": data.pop(self.text_key, ""),
            "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": data.pop("metadata", {}) | data,  # remaining data goes into metadata
        }

    def get_document_from_dict(self, data: dict, source_file: str, id_in_file: int | str):
        """
        Applies the adapter to each sample, instantiates a Document object and adds `default_metadata`.
        Args:
            data: a dictionary with the "raw" representation of the data
            source_file: file path or source for this sample
            id_in_file: its id in this particular file or source

        Returns: a Document

        """
        parsed_data = self.adapter(data, source_file, id_in_file)
        if not parsed_data.get("text", None):
            if not self._empty_warning:
                self._empty_warning = True
                logger.warning(
                    f"Found document without text, skipping. "
                    f'Is your `text_key` ("{self.text_key}") correct? Available keys: {list(data.keys())}'
                )
            return None
        document = Document(**parsed_data)
        if self.default_metadata:
            document.metadata = self.default_metadata | document.metadata
        return document

    @abstractmethod
    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        To be overridden
        """
        raise NotImplementedError


class BaseDiskReader(BaseReader):
    """Base module for fsspec based Readers. Readers read data from a source (local or remote) and create documents.

    Args:
        data_folder: the data folder to read from
        limit: limit the number of documents to read. Useful for debugging
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data from the source to a Document
        text_key: key to use for the text in the default adapter (default: "text"). Ignored if you provide your own `adapter`
        id_key: key to use for the id in the default adapter (default: "id"). Ignored if you provide your own `adapter`
        default_metadata: default metadata to add to all documents
        recursive: whether to read files recursively
        glob_pattern: glob pattern to filter files
    """

    type = "📖 - READER"

    def __init__(
        self,
        data_folder: DataFolderLike,
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        """

        Args:
            data_folder: a str, tuple or DataFolder object representing a path/filesystem
            limit: read at most this number of documents
            skip: skip the first n rows
            file_progress: show a tqdm progress bar for files
            doc_progress: show a tqdm progress bar for documents
            adapter: custom function that should return a dictionary with the datatrove Document format (see _default_adapter)
            text_key: the key containing the text data. `text` by default
            id_key: the key containing the id for each sample. `id` by default
            default_metadata: a dictionary with any data that should be added to all sample's metadata
            recursive: whether to search recursively for files
            glob_pattern: pattern that all files must match exactly to be included (relative to data_folder)
            shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use
            with dedup blocks
        """
        super().__init__(limit, skip, adapter, text_key, id_key, default_metadata)
        self.data_folder = get_datafolder(data_folder)
        self.recursive = recursive
        self.glob_pattern = glob_pattern
        self.shuffle_files = shuffle_files
        self.file_progress = file_progress
        self.doc_progress = doc_progress

    def get_document_from_dict(self, data: dict, source_file: str, id_in_file: int):
        document = super().get_document_from_dict(data, source_file, id_in_file)
        if document:
            document.metadata.setdefault("file_path", self.data_folder.resolve_paths(source_file))
        return document

    @abstractmethod
    def read_file(self, filepath: str) -> DocumentsPipeline:
        """
        Subclasses only need to implement this method. Should open the filepath given, and for each line/item in the file
         call `self.get_document_from_dict(data, filepath, id_in_path)` and yield its result.
        Args:
            filepath: path of the file to read

        Returns: generator of Document

        """
        raise NotImplementedError

    def read_files_shard(self, shard: list[str]) -> DocumentsPipeline:
        """
            Reads a list of files and yield Documents
        Args:
            shard: a list of file paths

        Returns: generator of Document

        """
        li = 0
        skipped = 0
        with (
            tqdm(
                total=self.limit if self.limit != -1 else None,
                desc="Document progress",
                unit="doc",
                disable=not self.doc_progress,
            ) as doc_pbar,
            tqdm(total=len(shard), desc="File progress", unit="file", disable=not self.file_progress) as file_pbar,
        ):
            for i, filepath in enumerate(shard):
                self.stat_update("input_files")
                logger.info(f"Reading input file {filepath}, {i+1}/{len(shard)}")
                di = 0
                for di, document in enumerate(self.read_file(filepath)):
                    if skipped < self.skip:
                        skipped += 1
                        continue
                    if self.limit != -1 and li >= self.limit:
                        break
                    yield document
                    doc_pbar.update()
                    li += 1
                file_pbar.update()
                self.stat_update("documents", value=di, unit="input_file")
                if self.limit != -1 and li >= self.limit:
                    break

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        Will get this rank's shard and sequentially read each file in the shard, yielding Document.
        Args:
            data: any existing data from previous pipeline stages
            rank: rank of the current task
            world_size: total number of tasks

        Returns:

        """
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
        if self.shuffle_files:
            random.shuffle(files_shard)
        for doc in self.read_files_shard(files_shard):
            self.update_doc_stats(doc)
            yield doc
