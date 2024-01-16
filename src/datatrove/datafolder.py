from typing import IO, TypeAlias

from fsspec import AbstractFileSystem
from fsspec import open as fsspec_open
from fsspec.core import url_to_fs
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem


class OutputFileManager:
    def __init__(self, fs, mode: str = "wt", compression: str | None = "infer"):
        self.fs = fs
        self.mode = mode
        self.compression = compression
        self._output_files = {}

    def get_file(self, filename):
        if filename not in self._output_files:
            self._output_files[filename] = self.fs.open(filename, mode=self.mode, compression=self.compression)
        return self._output_files[filename]

    def write(self, filename, data):
        self.get_file(filename).write(data)

    def __enter__(self):
        return self

    def close(self):
        for file in self._output_files.values():
            file.close()
        self._output_files.clear()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DataFolder(DirFileSystem):
    def __init__(
        self,
        path: str,
        fs: AbstractFileSystem | None = None,
        recursive: bool = True,
        glob: str | None = None,
        **storage_options,
    ):
        """
            `recursive` and `glob` will only affect the listing operation
        Args:
            path:
            fs:
            recursive:
            glob:
            **storage_options:
        """
        super().__init__(path=path, fs=fs if fs else url_to_fs(path, **storage_options)[0])
        self.recursive = recursive
        self.pattern = glob
        if not self.isdir("/"):
            self.mkdirs("/", exist_ok=True)

    def list_files(self, suffix: str = "", extension: str | list[str] = None) -> list[str]:
        if extension and isinstance(extension, str):
            extension = [extension]
        return sorted(
            [
                f
                for f in (
                    self.find(suffix, maxdepth=0 if not self.recursive else None)
                    if not self.pattern
                    else self.glob(
                        self.fs.sep.join([self.pattern, suffix]), maxdepth=0 if not self.recursive else None
                    )
                )
                if not extension or any(f.endswith(ext) for ext in extension)
            ]
        )

    def get_shard(self, rank: int, world_size: int, **kwargs) -> list[str]:
        """Fetch a shard (set of files) for a given rank, assuming there are a total of `world_size` shards.
        This should be deterministic to not have any overlap among different ranks.

        Args:
            rank: rank of the shard to fetch
            world_size: total number of shards
            other parameters will be passed to list_files

        Returns:
            a list of file paths
        """
        return self.list_files(**kwargs)[rank::world_size]

    def unstrip_protocol(self, name: str) -> str:
        if isinstance(self.fs, LocalFileSystem):
            return name
        return super().unstrip_protocol(name)

    def to_absolute_paths(self, paths) -> list[str] | str:
        if isinstance(paths, str):
            return self.unstrip_protocol(self._join(paths))
        return list(map(self.unstrip_protocol, self._join(paths)))

    def get_outputfile_manager(self, **kwargs) -> OutputFileManager:
        return OutputFileManager(self, **kwargs)

    def bulk_open_files(self, paths, mode="rb", **kwargs):
        return [self.open(path, mode=mode, **kwargs) for path in paths]


def get_datafolder(data: DataFolder | str) -> DataFolder | None:
    if data is None:
        return None
    if isinstance(data, str):
        return DataFolder(data)
    if isinstance(data, DataFolder):
        return data
    if isinstance(data, tuple) and list(map(type, data)) == [str, dict]:
        return DataFolder(data[0], **data[1])
    raise ValueError("You must pass a DataFolder or a str path")


def get_file(file: IO | str, mode="rt", **kwargs):
    if isinstance(file, str):
        return fsspec_open(file, mode, **kwargs)
    return file


ParsableDataFolder: TypeAlias = str | tuple[str, dict] | DataFolder
