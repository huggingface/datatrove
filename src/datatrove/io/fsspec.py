import os.path
from contextlib import contextmanager
from dataclasses import dataclass

import fsspec

from datatrove.io import InputDataFile
from datatrove.io.base import BaseInputDataFolder, BaseOutputDataFolder, OutputDataFile


@dataclass
class FSSpecOutputDataFile(OutputDataFile):
    _fs: fsspec.AbstractFileSystem = None

    def _create_file_handler(self, mode: str = "w", gzip: bool = False):
        assert gzip is False
        return self._fs.open(self.local_path, mode)


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
            self.unchecked_get_file(os.path.relpath(path, self.path))
            for path in self._fs.ls(os.path.join(self.path, suffix), detail=False)
            if self._match_file(path, extension)
        ]

    def unchecked_get_file(self, relative_path: str):
        return FSSpecInputDataFile(
            path=os.path.join(self.path, relative_path), relative_path=relative_path, _fs=self._fs
        )

    def file_exists(self, relative_path: str) -> bool:
        return self._fs.isfile(os.path.join(self.path, relative_path))
