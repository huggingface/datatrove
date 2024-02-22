import os.path
from glob import has_magic
from typing import IO, TypeAlias

from fsspec import AbstractFileSystem
from fsspec import open as fsspec_open
from fsspec.callbacks import NoOpCallback, TqdmCallback
from fsspec.core import get_fs_token_paths, url_to_fs
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem


class OutputFileManager:
    """
    Will keep track of different output files by name and properly cleanup in the end.
    """

    def __init__(self, fs, mode: str = "wt", compression: str | None = "infer"):
        self.fs = fs
        self.mode = mode
        self.compression = compression
        self._output_files = {}

    def get_file(self, filename):
        """
            Opens file `filename` if it hasn't been opened yet. Otherwise, just returns it from the file cache
        Args:
          filename: name of the file to open/get if previously opened

        Returns: a file handler we can write to

        """
        if filename not in self._output_files:
            self._output_files[filename] = self.fs.open(filename, mode=self.mode, compression=self.compression)
        return self._output_files[filename]

    def pop(self, filename):
        """
            Return the file, as if called by `get_file`, but clean up internal references to it.
        Args:
            filename: name of the file to open/get if previously opened

        Returns: a file handler we can write to
        """
        file = self.get_file(filename)
        self._output_files.pop(file)
        return file

    def write(self, filename, data):
        """
            Write data to a given file.
        Args:
          filename: file to be written
          data: data to write

        """
        self.get_file(filename).write(data)

    def __enter__(self):
        return self

    def close(self):
        """
        Close all currently open output files and clear the list of file.
        """
        for file in self._output_files.values():
            file.close()
        self._output_files.clear()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DataFolder(DirFileSystem):
    """
    Wrapper around a fsspec filesystem. All file operations will be relative to `path`.
    """

    def __init__(
        self,
        path: str,
        fs: AbstractFileSystem | None = None,
        auto_mkdir: bool = True,
        **storage_options,
    ):
        """
        Objects can be initialized with `path`, `path` and `fs` or `path` and `storage_options`
        Args:
            path: main path to base directory
            fs: fsspec filesystem to wrap
            auto_mkdir: if True, when opening a file in write mode its parent directories will be automatically created
            **storage_options: will be passed to a new fsspec filesystem object, when it is created. Ignored if fs is given
        """
        super().__init__(path=path, fs=fs if fs else url_to_fs(path, **storage_options)[0])
        self.auto_mkdir = auto_mkdir

    def list_files(
        self,
        subdirectory: str = "",
        recursive: bool = True,
        glob_pattern: str | None = None,
    ) -> list[str]:
        """
        Get a list of files on this directory. If `subdirectory` is given will search in `path/subdirectory`. If
        glob_pattern is given, it will only return files that match the pattern, which can be used to match a given
        extension, for example `*.myext`. Be careful with subdirectories when using glob (use ** if you want to match
        any subpath). Args: subdirectory: str:  (Default value = "") recursive: bool:  (Default value = True)
        glob_pattern: str | None:  (Default value = None)

        Returns: a list of file paths, relative to `self.path`

        """
        if glob_pattern and not has_magic(glob_pattern):
            # makes it slightly easier for file extensions
            glob_pattern = f"*{glob_pattern}"
        return sorted(
            [
                f
                for f, info in (
                    self.find(subdirectory, maxdepth=0 if not recursive else None, detail=True)
                    if not glob_pattern
                    else self.glob(
                        self.fs.sep.join([glob_pattern, subdirectory]),
                        maxdepth=0 if not recursive else None,
                        detail=True,
                    )
                ).items()
                if info["type"] != "directory"
            ]
        )

    def get_shard(self, rank: int, world_size: int, **kwargs) -> list[str]:
        """Fetch a shard (set of files) for a given rank, assuming there are a total of `world_size` shards.
        This should be deterministic to not have any overlap among different ranks.
        Will return files [rank, rank+world_size, rank+2*world_size, ...]
        Args:
          rank: int: rank of the shard to fetch
          world_size: int: total number of shards
          **kwargs:
        other parameters will be passed to list_files

        Returns: a list of file paths

        """
        return self.list_files(**kwargs)[rank::world_size]

    def resolve_paths(self, paths) -> list[str] | str:
        """
            Transform  a list of relative paths into a list of complete paths (including fs protocol and base path)
        Args:
          paths: list of relative paths

        Returns: list of fully resolved paths

        """
        if isinstance(paths, str):
            if isinstance(self.fs, LocalFileSystem):
                # make sure we strip file:// and similar
                return self.fs._strip_protocol(self._join(paths))
            # otherwise explicitly add back the protocol
            return self.fs.unstrip_protocol(self._join(paths))
        return list(map(self.resolve_paths, paths))

    def get_output_file_manager(self, **kwargs) -> OutputFileManager:
        """
            Factory for OutputFileManager
        Args:
          **kwargs: options to be passed to OutputFileManager

        Returns: new instance of OutputFileManager

        """
        return OutputFileManager(self, **kwargs)

    def open_files(self, paths, mode="rb", **kwargs):
        """
            Opens all files in an iterable with the given options, in the same order as given

        Args:
          paths: iterable of relative paths
          mode:  (Default value = "rb")
          **kwargs:

        Returns:

        """
        return [self.open(path, mode=mode, **kwargs) for path in paths]

    def open(self, path, mode="rb", *args, **kwargs):
        """
            Opens a single file.
            If self.auto_mkdir is `True`, will first make sure parent directories exist before opening in write mode.
        Args:
          path:
          mode:  (Default value = "rb")
          *args:
          **kwargs:

        Returns:

        """
        if self.auto_mkdir and ("w" in mode or "a" in mode):
            self.fs.makedirs(self.fs._parent(self._join(path)), exist_ok=True)
        return super().open(path, mode=mode, *args, **kwargs)


def get_datafolder(data: DataFolder | str | tuple[str, dict] | tuple[str, AbstractFileSystem]) -> DataFolder:
    """
    `DataFolder` factory.
    Possible input combinations:
    - `str`: the simplest way is to pass a single string. Example: `/home/user/mydir`, `s3://mybucket/myinputdata`,
    `hf://datasets/allenai/c4/en/`
    - `(str, fsspec filesystem instance)`: a string path and a fully initialized filesystem object.
    Example: `("s3://mybucket/myinputdata", S3FileSystem(client_kwargs={"endpoint_url": endpoint_uri}))`
    - `(str, dict)`: a string path and a dictionary with options to initialize a fs. Example
    (equivalent to the previous line): `("s3://mybucket/myinputdata", {"client_kwargs": {"endpoint_url": endpoint_uri}})`
    - `DataFolder`: you can initialize a DataFolder object directly and pass it as an argument


    Args:
      data: DataFolder | str | tuple[str, dict] | tuple[str, AbstractFileSystem]:

    Returns: `DataFolder` instance

    """
    # fully initialized DataFolder object
    if isinstance(data, DataFolder):
        return data
    # simple string path
    if isinstance(data, str):
        return DataFolder(data)
    # (str path, fs init options dict)
    if isinstance(data, tuple) and isinstance(data[0], str) and isinstance(data[1], dict):
        return DataFolder(data[0], **data[1])
    # (str path, initialized fs object)
    if isinstance(data, tuple) and isinstance(data[0], str) and isinstance(data[1], AbstractFileSystem):
        return DataFolder(data[0], fs=data[1])
    raise ValueError(
        "You must pass a DataFolder instance, a str path, a (str path, fs_init_kwargs) or (str path, fs object)"
    )


def open_file(file: IO | str, mode="rt", **kwargs):
    """

    Args:
      file: IO | str:
      mode:  (Default value = "rt")
      **kwargs:

    Returns:

    """
    if isinstance(file, str):
        return fsspec_open(file, mode, **kwargs)
    return file


def download_file(remote_path: str, local_path: str, progress: bool = True):
    fs, _, paths = get_fs_token_paths(remote_path)
    fs.get_file(
        paths[0],
        local_path,
        callback=TqdmCallback(
            tqdm_kwargs={
                "desc": f"↓ Downloading {os.path.basename(remote_path)}",
                "unit": "B",
                "unit_scale": True,
                "unit_divisor": 1024,  # make use of standard units e.g. KB, MB, etc.
                "miniters": 1,  # recommended for network progress that might vary strongly
            }
        )
        if progress
        else NoOpCallback(),
    )


DataFolderLike: TypeAlias = str | tuple[str, dict] | DataFolder
