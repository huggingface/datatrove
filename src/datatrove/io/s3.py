import gzip as gzip_lib
import os.path
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass

from loguru import logger

from datatrove.io import BaseInputDataFile, BaseOutputDataFolder
from datatrove.io.base import BaseInputDataFolder, BaseInputFileListFolder, BaseOutputDataFile
from datatrove.io.utils.s3 import (
    s3_download_file,
    s3_file_exists,
    s3_get_file_list,
    s3_get_file_stream,
    s3_upload_file,
)


@dataclass
class S3OutputDataFolder(BaseOutputDataFolder):
    """S3 output data folder. Output files will be uploaded to s3.
    Args:
        path (str): the s3 path to this folder
        local_path (str): where to save output data locally before uploading
        cleanup (str): remove local files after upload
    """

    local_path: str = None
    cleanup: bool = True

    def __post_init__(self):
        if not self.path.startswith("s3://"):
            raise ValueError("S3OutputDataFolder path must start with s3://")
        self._tmpdir = None
        if not self.local_path:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.local_path = self._tmpdir.name

    def close(self):
        super().close()
        if self._tmpdir:
            self._tmpdir.cleanup()

    def create_new_file(self, relative_path: str):
        return S3OutputDataFile(
            local_path=os.path.join(self.local_path, relative_path),
            path=os.path.join(self.path, relative_path),
            relative_path=relative_path,
            cleanup=self.cleanup,
            _lock=self._lock,
        )

    def to_input_folder(self) -> BaseInputDataFolder:
        return S3InputDataFolder(path=self.path, local_path=self.local_path, cleanup=self.cleanup)


@dataclass
class S3OutputDataFile(BaseOutputDataFile):
    """An individual S3OutputFile. This file is uploaded to s3 when closed."""

    local_path: str = None
    cleanup: bool = True

    def __post_init__(self):
        if not self.path.startswith("s3://"):
            raise ValueError("S3OutputDataFile path must start with s3://")
        assert self.local_path is not None, "S3OutputDataFile must have a local_path"

    def close(self):
        if self._file_handler:
            super().close()
            with self._lock:
                logger.info(f'Uploading "{self.local_path}" to "{self.path}"...')
                s3_upload_file(self.local_path, self.path)
                logger.info(f'Uploaded "{self.local_path}" to "{self.path}".')
            if self.cleanup:
                os.remove(self.local_path)

    def _create_file_handler(self, mode: str = "w", gzip: bool = False):
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        return open(self.local_path, mode) if not gzip else gzip_lib.open(self.local_path, mode)

    def delete(self):
        super().delete()
        if os.path.isfile(self.local_path):
            os.remove(self.local_path)


@dataclass
class S3InputDataFile(BaseInputDataFile):
    """An individual s3 input file.
    Args:
        local_path (str): local path where this file will be downloaded to (if `stream=False`)
        stream (bool): stream the file directly from s3, without saving to disk
    """

    local_path: str = None
    stream: bool = None
    cleanup: bool = True

    def __post_init__(self):
        if self.stream is None:
            self.stream = self.local_path is None

    @contextmanager
    def open_binary(self):
        if self.stream or self.local_path is None:
            # stream
            response_stream = s3_get_file_stream(self.path)
            try:
                yield response_stream
            finally:
                response_stream.close()
        else:
            # download
            if not os.path.isfile(self.local_path):
                with self._lock:
                    logger.info(f'Downloading "{self.path}" to "{self.local_path}"...')
                    s3_download_file(self.path, self.local_path)
                    logger.info(f'Downloaded "{self.path}" to "{self.local_path}".')
            with open(self.local_path, mode="rb") as f:
                try:
                    yield f
                finally:
                    if self.cleanup:
                        os.remove(self.local_path)


@dataclass
class S3InputDataFolder(BaseInputDataFolder):
    """S3 input data folder
    Args:
        local_path (str): where to download the files to (if `stream=False`)
        stream (bool): stream the file directly from s3, without saving to disk
        cleanup (str): remove downloaded files from disk after they are closed

    """

    local_path: str = None
    stream: bool = None
    cleanup: bool = True

    def __post_init__(self):
        if not self.path.startswith("s3://"):
            raise ValueError("S3InputDataFolder path must start with s3://")
        self._tmpdir = None
        if self.stream is None:
            self.stream = self.local_path is None
        if not self.local_path:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.local_path = self._tmpdir.name

    def _unchecked_get_file(self, relative_path: str):
        return S3InputDataFile(
            path=os.path.join(self.path, relative_path),
            local_path=os.path.join(self.local_path, relative_path),
            relative_path=relative_path,
            stream=self.stream,
            cleanup=self.cleanup,
            _lock=self._lock,
        )

    def file_exists(self, relative_path: str) -> bool:
        if self.local_path and os.path.exists(os.path.join(self.local_path, relative_path)):
            return True
        return s3_file_exists(os.path.join(self.path, relative_path))

    def list_files(self, extension: str | list[str] = None, suffix: str = "") -> list[BaseInputDataFile]:
        return [
            self._unchecked_get_file(os.path.join(suffix, path))
            for path in s3_get_file_list(
                os.path.join(self.path, suffix), match_pattern=self.match_pattern, recursive=self.recursive
            )
            if self._match_file(os.path.join(suffix, path), extension)
        ]

    def to_output_folder(self) -> BaseOutputDataFolder:
        return S3OutputDataFolder(path=self.path, local_path=self.local_path, cleanup=self.cleanup)


@dataclass
class S3InputFileListFolder(BaseInputFileListFolder):
    """S3 input file list folder
    Args:
        local_path (str): where to download the files to (if `stream=False`)
        stream (bool): stream the file directly from s3, without saving to disk
        cleanup (str): remove downloaded files from disk after they are closed

    """

    local_path: str = None
    stream: bool = None
    cleanup: bool = True

    def __post_init__(self):
        if not self.file_list[0].startswith("s3://"):
            raise ValueError("S3InputFileListFolder pathes must start with s3://")
        self._tmpdir = None
        if self.stream is None:
            self.stream = self.local_path is None
        if not self.local_path:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.local_path = self._tmpdir.name

    def _unchecked_get_file(self, file_path: str):
        return S3InputDataFile(
            path=file_path,
            local_path=os.path.join(self.local_path, file_path.replace("s3://", "")),
            relative_path=file_path.replace("s3://", ""),
            stream=self.stream,
            cleanup=self.cleanup,
            _lock=self._lock,
        )

    def file_exists(self, file_path: str) -> bool:
        if self.local_path and os.path.exists(os.path.join(self.local_path, file_path.replace("s3://", ""))):
            return True
        return s3_file_exists(file_path)

    def list_files(self) -> list[BaseInputDataFile]:
        return [self._unchecked_get_file(file_path) for file_path in self.file_list]
