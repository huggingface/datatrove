import random
from abc import abstractmethod
from types import MethodType
from typing import IO, Callable

from datatrove.utils.typeshelper import StatHints
from pyparsing import ABC
from tqdm import tqdm

from datatrove.data import Document, DocumentsPipeline, Media
from datatrove.io import DataFileLike, DataFolderLike, get_datafolder, get_shard_from_paths_file
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger


class BaseMediaReader(PipelineStep, ABC):
    """Base module for fsspec based Readers. Readers read data from a source (local or remote) and create documents.

    Args:
        data_folder: a str, tuple or DataFolder object representing a path/filesystem
        paths_file: optionally provide a file with one path per line (without the `data_folder` prefix) to read.
        limit: limit the number of documents to read. Useful for debugging
        skip: skip the first n rows
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Takes as input: (self, data: dict, path: str, id_in_file: int | str)
                self allows access to self.text_key and self.id_key
            Returns: a dict with at least a "text" and "id" keys
        text_key: the key containing the text data (default: "text").
        id_key: the key containing the id for each sample (default: "id").
        default_metadata: a dictionary with any data that should be added to all samples' metadata
        recursive: whether to search files recursively. Ignored if paths_file is provided
        glob_pattern: pattern that all files must match exactly to be included (relative to data_folder). Ignored if paths_file is provided
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use with dedup blocks
    """

    type = "📖 - READER"

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def read_media(self, media: Media) -> tuple[bytes | None, dict | None]:
        """
        Subclasses only need to implement this method. Should open the filepath given, and for each line/item in the file
         call `self.get_document_from_dict(data, filepath, id_in_path)` and yield its result.
        Args:
            media: Media object

        Returns: bytes of the media

        """
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        Will get this rank's shard and sequentially read each file in the shard, yielding Document.
        Args:
            data: any existing data from previous pipeline stages
            rank: rank of the current task
            world_size: total number of tasks

        Returns:

        """
        with self:
            for doc in data:
                with self.track_time():
                    for media in doc.media:
                        try:
                            media.media_bytes, metadata = self.read_media(media)
                        except Exception as e:
                            logger.error(f"Error reading media {media.id}: {e}")
                            continue
                        if metadata is not None:
                            media.metadata.update(metadata)
                        self.stat_update(StatHints.total)
                        self.update_media_stats(media)
                if not all(media.media_bytes is not None for media in doc.media):
                    logger.warning(f"Document {doc.id} has no media bytes, skipping")
                    continue
                yield doc


class SeekableMediaReader(BaseMediaReader, ABC):
    def __init__(self, data_folder: DataFolderLike):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.current_fp = None
        self.current_path = None

    @abstractmethod
    def open_file(self, path: str) -> IO:
        raise NotImplementedError

    @abstractmethod
    def close_file(self, fp: IO):
        raise NotImplementedError

    @abstractmethod
    def read_from_fp(self, fp: IO, media: Media) -> tuple[bytes | None, dict | None]:
        raise NotImplementedError

    def read_media(self, media: Media) -> tuple[bytes | None, dict | None]:
        if media.path is None:
            logger.warning(f"Media {media.id} has no path, skipping")
            return None, None

        if self.current_fp is None or self.current_path != media.path:
            if self.current_fp:
                self.close_file(self.current_fp)

            self.current_fp = self.open_file(media.path)
            self.current_path = media.path

        return self.read_from_fp(self.current_fp, media)

    def close(self):
        if self.current_fp:
            self.close_file(self.current_fp)


