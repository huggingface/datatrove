import os.path
from contextlib import contextmanager
from dataclasses import dataclass

import fsspec

from datatrove.io import InputDataFile
from datatrove.io.base import BaseInputDataFolder, BaseOutputDataFolder, OutputDataFile


@dataclass
class FSSpecOutputDataFile(OutputDataFile):
    _fs: fsspec.AbstractFileSystem = None

    def open(self, mode: str = "w", gzip: bool = False, overwrite: bool = False):
        assert gzip is False
        if not self.file_handler or overwrite:
            self.file_handler = self._fs.open(self.local_path, mode)
        return self


@dataclass
class FSSpecOutputDataFolder(BaseOutputDataFolder):
    storage_options: dict = None

    def __post_init__(self):
        super().__post_init__()
        protocol, self.path = self.path.split("://")
        self._fs = fsspec.filesystem(protocol, **(self.storage_options if self.storage_options else {}))

    def create_new_file(self, relative_path: str):
        return FSSpecOutputDataFile(
            path=os.path.join(self.path, relative_path), relative_path=relative_path, _fs=self._fs, local_path=None
        )


@dataclass
class FSSpecInputDataFile(InputDataFile):
    _fs: fsspec.AbstractFileSystem

    @contextmanager
    def open_binary(self):
        with self._fs.open(self.path, mode="rb") as f:
            yield f


@dataclass
class FSSpecInputDataFolder(BaseInputDataFolder):
    storage_options: dict = None

    def __post_init__(self):
        super().__post_init__()
        if "://" in self.path:
            protocol, self.path = self.path.split("://")
        else:
            protocol = "file"
        self._fs = fsspec.filesystem(protocol, **(self.storage_options if self.storage_options else {}))

    def list_files(self, extension: str | list[str] = None, suffix: str = "") -> list[InputDataFile]:
        return [
            FSSpecInputDataFile(path=path, relative_path=os.path.relpath(path, self.path), _fs=self._fs)
            for path in self._fs.ls(self.path, detail=False)
            if self._match_file(path, extension)
        ]
