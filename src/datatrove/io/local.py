import gzip as gzip_lib
import os
from contextlib import contextmanager
from dataclasses import dataclass

from datatrove.io.base import BaseInputDataFile, BaseInputDataFolder, BaseOutputDataFile, BaseOutputDataFolder


@dataclass
class LocalOutputDataFile(BaseOutputDataFile):
    def _create_file_handler(self, mode: str = "w", gzip: bool = False):
        """
            Opens/creates the underlying file object
        :param mode:
        :param gzip:
        :return:
        """
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        return open(self.path, mode) if not gzip else gzip_lib.open(self.path, mode)

    def delete(self):
        super().delete()
        if os.path.isfile(self.path):
            os.remove(self.path)

    @property
    def persistent_local_path(self):
        return self.path


@dataclass
class LocalOutputDataFolder(BaseOutputDataFolder):
    """
    Local output data folder. The output files will simply be saved to a folder on disk.
    Args:
        path (str): absolute path to a local folder on disk
    """

    def _create_new_file(self, relative_path: str) -> LocalOutputDataFile:
        return LocalOutputDataFile(
            path=os.path.join(self.path, relative_path), relative_path=relative_path, _lock=self._lock
        )

    def to_input_folder(self) -> BaseInputDataFolder:
        return LocalInputDataFolder(path=self.path)


@dataclass
class LocalInputDataFile(BaseInputDataFile):
    @contextmanager
    def open_binary(self):
        """
            Opens local file in binary mode
        :return: binary stream
        """
        with open(self.path, mode="rb") as f:
            yield f


@dataclass
class LocalInputDataFolder(BaseInputDataFolder):
    """
    Local input data folder. Read data already present on disk.
    Args:
        path (str): absolute path to a local folder on disk

    """

    def list_files(self, extension: str | list[str] = None, suffix: str = "") -> list[LocalInputDataFile]:
        return [
            self._unchecked_get_file(os.path.relpath(path, self.path))
            for path in get_local_file_list(os.path.join(self.path, suffix), self.recursive)
            if self._match_file(path, extension)
        ]

    def file_exists(self, relative_path: str) -> bool:
        return os.path.exists(os.path.join(self.path, relative_path))

    def _unchecked_get_file(self, relative_path: str) -> LocalInputDataFile:
        """
            Get a file directly by its name, without checking if it exists.
        :param relative_path: The file name/path from this folder.
        :return: an input file
        """
        return LocalInputDataFile(
            path=os.path.join(self.path, relative_path), relative_path=relative_path, _lock=self._lock
        )


def get_local_file_list(path: str, recursive: bool = True) -> list[str]:
    """
        Get a list of absolute paths to all the files in a given local folder, sorted.
    :param path: The path to scan
    :param recursive: if False will only scan the top level files (no subdirectories)
    :return: list of file paths
    """
    filelist = []
    with os.scandir(path) as files:
        for f in files:
            if f.is_file():
                filelist.append(os.path.abspath(f.path))
            elif recursive:
                filelist.extend(get_local_file_list(f.path, recursive=recursive))
    return sorted(filelist)
