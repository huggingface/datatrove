import contextlib
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from fnmatch import fnmatch
from gzip import GzipFile
from io import TextIOWrapper
from typing import Literal

import numpy as np
import zstandard
from loguru import logger

from datatrove.io.utils.fsspec import valid_fsspec_path


@dataclass
class BaseInputDataFile:
    """
    Represents an individual input file that can be read from.
    Supports compression formats and/or opening in binary mode.

    Args:
        path (str): complete path to this input file
        relative_path (str): path relative to the input data folder this file belongs to
    """

    path: str
    relative_path: str
    folder: "BaseInputDataFolder"

    @contextmanager
    @abstractmethod
    def open_binary(self):
        """
            Main method to overwrite. Should return a stream open in binary mode
        :return: binary stream
        """
        raise NotImplementedError

    @contextmanager
    def open_gzip(self, binary=False):
        """
            Read a gzip compressed file directly
        :param binary: optionally open in binary mode
        :return: stream
        """
        with self.open_binary() as fo:
            with GzipFile(mode="r" if not binary else "rb", fileobj=fo) as gf:
                if binary:
                    yield gf
                else:
                    with TextIOWrapper(gf) as f:
                        yield f

    @contextmanager
    def open_zst(self, binary=False):
        """
            Read a zst compressed file directly
        :param binary:
        :return: stream
        """
        with self.open_binary() as fo:
            dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
            stream_reader = dctx.stream_reader(fo)
            if binary:
                yield stream_reader
            else:
                with TextIOWrapper(stream_reader) as f:
                    yield f

    @contextmanager
    def open(self, binary=False, compression: Literal["guess", "gzip", "zst"] | None = None):
        """
            Main entrypoint for input files. Return a stream to read this file. Optionally accepts compression and/or binary mode.
        :param binary:
        :param compression:
        :return:
        """
        if compression == "guess":
            compression = guess_compression(self.relative_path)
        match compression:
            case "gzip":
                with self.open_gzip(binary) as f:
                    yield f
            case "zst":
                with self.open_zst(binary) as f:
                    yield f
            case _:
                with self.open_binary() as fo:
                    if binary:
                        yield fo
                    else:
                        with TextIOWrapper(fo) as f:
                            yield f


@dataclass
class BaseInputDataFolder(ABC):
    """
        Base input data folder class. Specific implementations should override its relevant methods.

    Args:
        path (str): full path to the folder, respecting the format of specific implementations
            (i.e. s3 paths should start with s3://)
        extension (str | list[str], optional): file extensions to filter. Defaults to None.
        recursive (bool, optional): whether to search recursively. Defaults to True.
        match_pattern (str, optional): pattern to match file names. Defaults to None.
    """

    path: str
    extension: str | list[str] = None
    recursive: bool = True
    match_pattern: str = None
    _lock = contextlib.nullcontext()

    @classmethod
    def from_path(cls, path, **kwargs):
        """
            InputDataFolder factory: get the correct instance from a path. Additional arguments will be passed to the
            constructor of the matched implementation.
        :param path: the full path to match
        :param kwargs: any additional argument passed to the matched implementation
        :return:
        """
        from datatrove.io import FSSpecInputDataFolder, LocalInputDataFolder, S3InputDataFolder

        if path.startswith("s3://"):
            return S3InputDataFolder(path, **kwargs)
        elif valid_fsspec_path(path):
            return FSSpecInputDataFolder(path, **kwargs)
        return LocalInputDataFolder(path, **kwargs)

    @abstractmethod
    def list_files(self, extension: str | list[str] = None, suffix: str = "") -> list[BaseInputDataFile]:
        """
            Retrieves a list of InputDataFile for all files in this folder matching both the dataclass properties and the
            function parameters.
        :param extension: optionally limit the file extension of the retrieved files
        :param suffix: optionally limit the search to a given subfolder
        :return:
        """
        logger.error(
            "Do not instantiate BaseInputDataFolder directly, "
            "use a LocalInputDataFolder, S3InputDataFolder or call"
            "BaseInputDataFolder.from_path(path)"
        )
        raise NotImplementedError

    def set_lock(self, lock):
        """
            Pass a synchronization primitive to limit concurrent downloads
        :param lock: synchronization primitive (Semaphore, Lock)
        :return:
        """
        self._lock = lock

    def get_files_shard(
        self, rank: int, world_size: int, extension: str | list[str] = None
    ) -> list[BaseInputDataFile]:
        """
            Fetch a shard (set of files) for a given rank, assuming there are a total of `world_size` shards.
            This should be deterministic to not have any overlap among different ranks.

        :param rank: rank of the shard to fetch
        :param world_size: total number of shards
        :param extension: optional file extension to pass on to list_files
        :return: a list of input files
        """
        return self.list_files(extension=extension)[rank::world_size]

    def get_file(self, relative_path: str) -> BaseInputDataFile | None:
        """
            Get a file directly by its name.
        :param relative_path: The file name/path from this folder.
        :return: an input file if it exists or None
        """
        if self.file_exists(relative_path):
            return self._unchecked_get_file(relative_path)

    @abstractmethod
    def _unchecked_get_file(self, relative_path: str) -> BaseInputDataFile:
        """
            Get a file directly by its name, without checking if it exists.
            Subclasses should override this method (and not the one above).
            Should not be called directly. Instead, call `get_file`
        :param relative_path: The file name/path from this folder.
        :return: an input file
        """
        raise NotImplementedError

    @abstractmethod
    def file_exists(self, relative_path: str) -> bool:
        """
            Should be overriden by subclasses. Check if a given file exists.
        :param relative_path: The file name/path from this folder.
        :return: if the file exists
        """
        return True

    def _match_file(self, file_path, extension=None) -> bool:
        """
            Checks if a given file matches the chosen extension(s) and/or pattern.
        :param file_path: the relative file path
        :param extension:
        :return: bool
        """
        extensions = (
            ([self.extension] if isinstance(self.extension, str) else self.extension)
            if not extension
            else ([extension] if isinstance(extension, str) else extension)
        )
        return (
            not extensions or any((get_extension(file_path).endswith(ext) for ext in extensions))
        ) and (  # check extension  # check pattern
            not self.match_pattern or fnmatch(os.path.relpath(file_path, self.path), self.match_pattern)
        )


def get_extension(filepath, depth=None) -> str:
    """
        Get full file extension (example: .jsonl.gz)
        Optionally only get last `depth` extensions (depth=1: .gz)
    :param filepath: relative path to the file
    :param depth: how many extensions maximum to get
    :return: the extension
    """
    exts = []
    stem, ext = os.path.splitext(filepath)
    while ext:
        exts.append(ext)
        stem, ext = os.path.splitext(stem)
        if depth and len(exts) >= depth:
            break
    return "".join(reversed(exts))


def guess_compression(filename) -> str | None:
    """
        Guess compression scheme from file extension
    :param filename:
    :return:
    """
    match get_extension(filename, depth=1):
        case ".gz":
            return "gzip"
        case ".zst":
            return "zst"
        case _:
            return None


@dataclass
class BaseOutputDataFile(ABC):
    """
    Represents an individual output file that can be written to.
    Supports writing directly to a gzip compressed file

    Args:
        path (str): absolute path of this file in its original form (example: s3://...)
        relative_path (str): path relative to the output data folder this file belongs to
    """

    path: str
    relative_path: str
    folder: "BaseOutputDataFolder"
    _file_handler = None
    _mode: str = None

    def close(self):
        """
            Close the underlying file object.
            Subclasses my also save/upload the file
        :return:
        """
        if self._file_handler:
            self._file_handler.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self, mode: str = "w", gzip: bool = False):
        """
            Open/create the underlying file object
        :param mode: mode to open
        :param gzip: whether to compress the file with gzip
        :return:
        """
        if not self._file_handler or self._mode != mode:
            if self._file_handler:
                self._file_handler.close()
            self._file_handler = self._create_file_handler(mode, gzip)
            self._mode = mode
        return self

    @abstractmethod
    def _create_file_handler(self, mode: str = "w", gzip: bool = False):
        """
            Should be overriden by subclasses. Opens/creates the underlying file object
        :param mode:
        :param gzip:
        :return:
        """
        raise NotImplementedError

    def write(self, *args, **kwargs):
        """
            Write to the underlying file object
        :param args:
        :param kwargs:
        :return:
        """
        self._file_handler.write(*args, **kwargs)

    def get_mmap(self, dtype):
        """
            Get a numpy memmap for this file
        :param dtype: memmap dtype
        :return:
        """
        return np.memmap(self._file_handler, dtype=dtype)

    def __del__(self):
        """
            Remove pointer to the given file, close its underlying file object and delete it from disk if it exists locally.
            Does not call close() on the OutputDataFile directly to avoid uploading/saving it externally.
        :return:
        """
        self.folder.pop_file(self.relative_path)
        if self._file_handler:
            self._file_handler.close()


@dataclass
class BaseOutputDataFolder(ABC):
    """
        Base output data folder class. Specific implementations should override its relevant methods.

    Args:
        path (str): full path to the folder, respecting the format of specific implementations
            (i.e. s3 paths should start with s3://)
    """

    path: str
    _output_files: dict[str, BaseOutputDataFile] = field(default_factory=dict)
    _lock = contextlib.nullcontext()

    def close(self):
        """
        Close all the output files and cleanup any leftover temporary files.
        """
        for file in self._output_files.values():
            file.close()

    @classmethod
    def from_path(cls, path: str, **kwargs):
        """
            OutputDataFolder factory: get the correct instance from a path. Additional arguments will be passed to the
            constructor of the matched implementation.
        :param path: the full path to match
        :param kwargs: any additional argument passed to the matched implementation
        :return:
        """
        from datatrove.io import FSSpecOutputDataFolder, LocalOutputDataFolder, S3OutputDataFolder

        if path.startswith("s3://"):
            return S3OutputDataFolder(path, **kwargs)
        elif valid_fsspec_path(path):
            return FSSpecOutputDataFolder(path, **kwargs)
        return LocalOutputDataFolder(path, **kwargs)

    @abstractmethod
    def create_new_file(self, relative_path: str) -> BaseOutputDataFile:
        """
            Create an OutputDataFile for the file located on `relative_path`, its path relative to this folder.
            Each io implementation should override it.
        :param relative_path:
        :return: an OutputDataFile
        """
        logger.error(
            "Do not instantiate a BaseOutputDataFolder directly, " "use a LocalOutputDataFolder or S3OutputDataFolder"
        )
        raise NotImplementedError

    def set_lock(self, lock):
        """
            Pass a synchronization primitive to limit concurrent uploads
        :param lock: synchronization primitive (Semaphore, Lock)
        :return:
        """
        self._lock = lock

    def delete_file(self, relative_path: str):
        """
            Remove pointer to the given file, close its underlying file object and delete it from disk if it exists locally.
            Does not call close() on the OutputDataFile directly to avoid uploading/saving it externally.
        :param relative_path: relative_path to the file
        :return:
        """
        if relative_path in self._output_files:
            output_file = self._output_files.pop(relative_path)
            # only close the actual file_handler, avoid any sort of upload
            if output_file._file_handler:
                output_file._file_handler.close()
            if output_file.local_path and os.path.isfile(output_file.local_path):
                os.remove(output_file.local_path)

    def open(self, relative_path: str, mode: str = "w", gzip: bool = False) -> BaseOutputDataFile:
        """
            Open/create output file with `relative_path` for writing.
        :param relative_path: the file path relative to this folder
        :param mode: the mode to open as (w, wb, etc)
        :param gzip: whether to open in gzip mode
        :return: OutputDataFile
        """
        if relative_path not in self._output_files:
            self._output_files[relative_path] = self.create_new_file(relative_path)
        return self._output_files[relative_path].open(mode, gzip)

    def pop_file(self, relative_path):
        if relative_path in self._output_files:
            self._output_files.pop(relative_path)
