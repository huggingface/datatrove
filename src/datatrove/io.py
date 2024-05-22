import os.path
from glob import has_magic
from typing import IO, Callable, TypeAlias

from fsspec import AbstractFileSystem
from fsspec import open as fsspec_open
from fsspec.callbacks import NoOpCallback, TqdmCallback
from fsspec.core import get_fs_token_paths, strip_protocol, url_to_fs
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfFileSystem, cached_assets_path

from datatrove.utils.logging import logger


class OutputFileManager:
    """A simple file manager to create/handle/close multiple output files.
        Will keep track of different output files by name and properly cleanup in the end.

    Args:
        fs: the filesystem to use (see fsspec for more details)
        mode: the mode to open the files with
        compression: the compression to use
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

    def get_open_files(self):
        """
        Getter for output files
        """
        return self._output_files

    def pop(self, filename):
        """
            Return the file, as if called by `get_file`, but clean up internal references to it.
        Args:
            filename: name of the file to open/get if previously opened

        Returns: a file handler we can write to
        """
        file = self.get_file(filename)
        self._output_files.pop(filename)
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
    """A simple wrapper around fsspec's DirFileSystem to handle file listing and sharding files accross multiple workers/process.
        Also handles the creation of output files.
        All file operations will be relative to `path`.

    Args:
        path: the path to the folder (local or remote)
        fs: the filesystem to use (see fsspec for more details)
        auto_mkdir: whether to automatically create the parent directories when opening a file in write mode
        **storage_options: additional options to pass to the filesystem
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
        include_directories: bool = False,
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
        extra_options = {}
        if isinstance(self.fs, HfFileSystem):
            extra_options["expand_info"] = False  # speed up
        if include_directories:
            extra_options["withdirs"] = True
        return sorted(
            [
                f
                for f, info in (
                    self.find(subdirectory, maxdepth=1 if not recursive else None, detail=True, **extra_options)
                    if not glob_pattern
                    else self.glob(
                        self.fs.sep.join([subdirectory, glob_pattern]) if subdirectory else glob_pattern,
                        maxdepth=1 if not recursive else None,
                        detail=True,
                        **extra_options,
                    )
                ).items()
                if include_directories or info["type"] != "directory"
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
        """Opens all files in an iterable with the given options, in the same order as given

        Args:
            paths: iterable of relative paths
            mode: the mode to open the files with (Default value = "rb")
            **kwargs: additional arguments to pass to the open
        """
        return [self.open(path, mode=mode, **kwargs) for path in paths]

    def open(self, path, mode="rb", *args, **kwargs):
        """Open a file locally or remote, and create the parent directories if self.auto_mkdir is `True` and we are opening in write mode.

            args/kwargs will depend on the filesystem (see fsspec for more details)
            Typically we often use:
                - compression: the compression to use
                - block_size: the block size to use

        Args:
            path: the path to the file
            mode: the mode to open the file with (Default value = "rb")
            *args: additional arguments to pass to the open
            **kwargs: additional arguments to pass to the open
        """
        if self.auto_mkdir and ("w" in mode or "a" in mode):
            self.fs.makedirs(self.fs._parent(self._join(path)), exist_ok=True)
        return super().open(path, mode=mode, *args, **kwargs)

    def is_local(self):
        """
        Checks if the underlying fs instance is a LocalFileSystem
        """
        return isinstance(self.fs, LocalFileSystem)


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
    """Wrapper around fsspec.open to handle both file-like objects and string paths

    Args:
      file: IO | str:
      mode:  (Default value = "rt")
      **kwargs:

    Returns:
    """
    if isinstance(file, str):
        return fsspec_open(file, mode, **kwargs)
    return file


def file_exists(path: str):
    fs, a, fpath = get_fs_token_paths(path)
    return fs.exists(fpath[0])


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


def safely_create_file(file_to_lock: str, do_processing: Callable):
    """
    Gets a lock to download/process and create some file(s). When processing is done a ".completed" file is created.
    If this file already exists, we skip the processing. Otherwise, we try to acquire a lock and when we get it if the
    completed file has not been created yet, we run the processing.

    Args:
        file_to_lock: str: lock will be "lock_path.lock" and completed file "lock_path.completed"
        do_processing: callback with the code to run to process/create the files
    """
    from fasteners import InterProcessLock

    completed_file = f"{file_to_lock}.completed"

    # if the completed file exists, we exit straight away
    if os.path.exists(completed_file):
        return

    # file is either being downloaded or needs to be downloaded
    with InterProcessLock(f"{file_to_lock}.lock"):
        if not os.path.exists(completed_file):
            do_processing()
            open(completed_file, "a").close()


def cached_asset_path_or_download(
    remote_path: str, progress: bool = True, namespace: str = "default", subfolder: str = "default", desc: str = "file"
):
    """
    Download a file from a remote path to a local path.
    This function is process-safe and will only download the file if it hasn't been downloaded already.
    Args:
        namespace: will group diff blocks. example: "filters"
        subfolder: relative to the specific block calling this function. Example: "language_filter"
        remote_path: str: The remote path to the file to download
        progress: bool: Whether to show a progress bar (Default value = True)
        desc: description of the file being downloaded
    """

    download_dir = cached_assets_path(library_name="datatrove", namespace=namespace, subfolder=subfolder)
    local_path = os.path.join(download_dir, strip_protocol(remote_path).replace("/", "_"))

    def do_download_file():
        logger.info(f'⬇️ Downloading {desc} from "{remote_path}"...')
        download_file(remote_path, local_path, progress)
        logger.info(f'⬇️ Downloaded {desc} to "{local_path}".')

    safely_create_file(local_path, do_download_file)
    return local_path


DataFolderLike: TypeAlias = str | tuple[str, dict] | DataFolder
