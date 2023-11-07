import contextlib
import gzip as gzip_lib
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from fnmatch import fnmatch
from gzip import GzipFile
from io import TextIOWrapper
from typing import Literal

import zstandard
from loguru import logger

from datatrove.io.utils.fsspec import valid_fsspec_path


@dataclass
class InputDataFile:
    path: str
    relative_path: str

    @contextmanager
    def open_binary(self):
        with open(self.path, mode="rb") as f:
            yield f

    @contextmanager
    def open_gzip(self, binary=False):
        with self.open_binary() as fo:
            with GzipFile(mode="r" if not binary else "rb", fileobj=fo) as gf:
                if binary:
                    yield gf
                else:
                    with TextIOWrapper(gf) as f:
                        yield f

    @contextmanager
    def open_zst(self, binary=False):
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
    """An input data folder

    Args:
        path (str): path to the folder
        extension (str | list[str], optional): file extensions to filter. Defaults to None.
        recursive (bool, optional): whether to search recursively. Defaults to True.
        match_pattern (str, optional): pattern to match file names. Defaults to None.
    """

    path: str
    extension: str | list[str] = None
    recursive: bool = True
    match_pattern: str = None

    @classmethod
    def from_path(cls, path, **kwargs):
        from datatrove.io import FSSpecInputDataFolder, LocalInputDataFolder, S3InputDataFolder

        if path.startswith("s3://"):
            return S3InputDataFolder(path, **kwargs)
        elif valid_fsspec_path(path):
            return FSSpecInputDataFolder(path, **kwargs)
        return LocalInputDataFolder(path, **kwargs)

    @abstractmethod
    def list_files(self, extension: str | list[str] = None, suffix: str = "") -> list[InputDataFile]:
        logger.error(
            "Do not instantiate BaseInputDataFolder directly, "
            "use a LocalInputDataFolder, S3InputDataFolder or call"
            "BaseInputDataFolder.from_path(path)"
        )
        raise NotImplementedError

    def __post_init__(self):
        self._lock = contextlib.nullcontext()

    def set_lock(self, lock):
        self._lock = lock

    def get_files_shard(self, rank: int, world_size: int, extension: str | list[str] = None) -> list[InputDataFile]:
        return self.list_files(extension=extension)[rank::world_size]

    def get_file(self, relative_path: str) -> InputDataFile | None:
        if self.file_exists(relative_path):
            return self.unchecked_get_file(relative_path)

    def unchecked_get_file(self, relative_path: str) -> InputDataFile:
        return InputDataFile(path=os.path.join(self.path, relative_path), relative_path=relative_path)

    @abstractmethod
    def file_exists(self, relative_path: str) -> bool:
        return True

    def _match_file(self, file_path, extension=None):
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


def get_extension(filepath, depth=None):
    exts = []
    stem, ext = os.path.splitext(filepath)
    while ext:
        exts.append(ext)
        stem, ext = os.path.splitext(stem)
        if depth and len(exts) >= depth:
            break
    return "".join(reversed(exts))


def guess_compression(filename):
    match get_extension(filename, depth=1):
        case ".gz":
            return "gzip"
        case ".zst":
            return "zst"
        case _:
            return None


@dataclass
class OutputDataFile(ABC):
    local_path: str | None
    path: str
    relative_path: str
    file_handler = None
    nr_documents: int = 0
    mode: str = None

    def close(self):
        if self.file_handler:
            self.file_handler.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self, mode: str = "w", gzip: bool = False):
        if not self.file_handler or self.mode != mode:
            if self.file_handler:
                self.file_handler.close()
            self.file_handler = self._create_file_handler(mode, gzip)
            self.mode = mode
        return self

    def _create_file_handler(self, mode: str = "w", gzip: bool = False):
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        return open(self.local_path, mode) if not gzip else gzip_lib.open(self.local_path, mode)

    def write(self, *args, **kwargs):
        self.file_handler.write(*args, **kwargs)


@dataclass
class BaseOutputDataFolder(ABC):
    path: str
    local_path: str
    _output_files: dict[str, OutputDataFile] = field(default_factory=dict)

    def close(self):
        for file in self._output_files.values():
            file.close()

    @classmethod
    def from_path(cls, path: str, **kwargs):
        from datatrove.io import FSSpecOutputDataFolder, LocalOutputDataFolder, S3OutputDataFolder

        if path.startswith("s3://"):
            return S3OutputDataFolder(path, **kwargs)
        elif valid_fsspec_path(path):
            return FSSpecOutputDataFolder(path, **kwargs)
        return LocalOutputDataFolder(path, **kwargs)

    @abstractmethod
    def create_new_file(self, relative_path: str) -> OutputDataFile:
        logger.error(
            "Do not instantiate a BaseOutputDataFolder directly, " "use a LocalOutputDataFolder or S3OutputDataFolder"
        )
        raise NotImplementedError

    def __post_init__(self):
        self._lock = contextlib.nullcontext()

    def set_lock(self, lock):
        self._lock = lock

    def delete_file(self, relative_path: str):
        if relative_path in self._output_files:
            output_file = self._output_files.pop(relative_path)
            output_file.close()
            if output_file.local_path and os.path.isfile(output_file.local_path):
                os.remove(output_file.local_path)

    def open(self, relative_path: str, mode: str = "w", gzip: bool = False):
        if relative_path not in self._output_files:
            self._output_files[relative_path] = self.create_new_file(relative_path)
        return self._output_files[relative_path].open(mode, gzip)
