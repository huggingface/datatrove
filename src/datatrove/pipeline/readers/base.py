from abc import abstractmethod
from contextlib import nullcontext
from typing import Callable

from loguru import logger
from tqdm import tqdm

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep


class BaseReader(PipelineStep):
    """Base module for Readers. Readers read data from a source and create documents.
        Reader are the first step in a pipeline usually.

    Args:
        limit: limit the number of documents to read. Useful for debugging
        progress: show tqdm progress bar. Might be spammy in some environments
        adapter: function to adapt the data dict from the source to a Document.
            Take as input: data: dict, path: str, id_in_file: int | str
            Return: a dict with at least a "text" key
        text_key: key to use for the text in the default adapter (default: "text"). Ignored if you provide your own `adapter`
        id_key: key to use for the id in the default adapter (default: "id"). Ignored if you provide your own `adapter`
        default_metadata: default metadata to add to all documents
    """

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
        """

        Args:
            limit: read at most this number of documents
            progress: show a tqdm progress bar
            adapter: custom function that should return a dictionary with the datatrove Document format (see _default_adapter)
            text_key: the key containing the text data. `text` by default
            id_key: the key containing the id for each sample. `id` by default
            default_metadata: a dictionary with any data that should be added to all sample's metadata
        """
        super().__init__()
        self.limit = limit
        self.progress = progress
        self.text_key = text_key
        self.id_key = id_key
        self.adapter = adapter if adapter else self._default_adapter
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
                    f"Found document without text, skipping. " f'Is your `text_key` ("{self.text_key}") correct?'
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
        progress: show progress bar
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
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
    ):
        """

        Args:
            data_folder: a str, tuple or DataFolder object representing a path/filesystem
            limit: read at most this number of documents
            progress: show a tqdm progress bar
            adapter: custom function that should return a dictionary with the datatrove Document format (see _default_adapter)
            text_key: the key containing the text data. `text` by default
            id_key: the key containing the id for each sample. `id` by default
            default_metadata: a dictionary with any data that should be added to all sample's metadata
            recursive: whether to search recursively for files
            glob_pattern: pattern that all files must match exactly to be included (relative to data_folder)
        """
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
        for doc in self.read_files_shard(files_shard):
            self.update_doc_stats(doc)
            yield doc
