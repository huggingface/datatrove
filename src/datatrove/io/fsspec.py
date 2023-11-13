import os.path
from contextlib import contextmanager
from dataclasses import dataclass

import fsspec

from datatrove.io import BaseInputDataFile
from datatrove.io.base import BaseInputDataFolder, BaseOutputDataFile, BaseOutputDataFolder


@dataclass
class FSSpecOutputDataFile(BaseOutputDataFile):
    """
    An individual FSSpecOutputDataFile.
    """

    _fs: fsspec.AbstractFileSystem = None

    def _create_file_handler(self, mode: str = "w", gzip: bool = False):
        assert gzip is False
        return self._fs.open(self.path, mode)

    def delete(self):
        super().delete()
        if self._fs.isfile(self.path):
            self._fs.rm(self.path)


@dataclass
class FSSpecOutputDataFolder(BaseOutputDataFolder):
    """
    fsspec output data folder. Accepts any valid fsspec path schema, such as hf://datasets/...
    Args:
        path (str): the fsspec path to this folder
        local_path (str): where to save output data locally before uploading
        cleanup (str): remove local files after upload
    """

    storage_options: dict = None

    def __post_init__(self):
        if "://" in self.path:
            protocol, self.path = self.path.split("://")
        else:
            protocol = "file"
        self._fs = fsspec.filesystem(protocol, **(self.storage_options if self.storage_options else {}))

    def _create_new_file(self, relative_path: str):
        return FSSpecOutputDataFile(
            path=os.path.join(self.path, relative_path), relative_path=relative_path, _fs=self._fs, _lock=self._lock
        )

    def to_input_folder(self) -> BaseInputDataFolder:
        return FSSpecInputDataFolder(path=self.path, storage_options=self.storage_options)


@dataclass
class FSSpecInputDataFile(BaseInputDataFile):
    _fs: fsspec.AbstractFileSystem = None

    @contextmanager
    def open_binary(self):
        with self._fs.open(self.path, mode="rb") as f:
            yield f


@dataclass
class FSSpecInputDataFolder(BaseInputDataFolder):
    storage_options: dict = None

    def __post_init__(self):
        if "://" in self.path:
            protocol, self.path = self.path.split("://")
        else:
            protocol = "file"
        self._fs = fsspec.filesystem(protocol, **(self.storage_options if self.storage_options else {}))

    def list_files(self, extension: str | list[str] = None, suffix: str = "") -> list[BaseInputDataFile]:
        return [
            self._unchecked_get_file(os.path.relpath(path, self.path))
            for path in self._fs.ls(os.path.join(self.path, suffix), detail=False)
            if self._match_file(path, extension)
        ]

    def _unchecked_get_file(self, relative_path: str):
        return FSSpecInputDataFile(
            path=os.path.join(self.path, relative_path), relative_path=relative_path, _fs=self._fs, _lock=self._lock
        )

    def file_exists(self, relative_path: str) -> bool:
        return self._fs.isfile(os.path.join(self.path, relative_path))
