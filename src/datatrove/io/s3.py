import gzip as gzip_lib
import os.path
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass

from loguru import logger

from datatrove.io import BaseOutputDataFolder, InputDataFile
from datatrove.io.base import BaseInputDataFolder, OutputDataFile
from datatrove.io.utils.s3 import (
    s3_download_file,
    s3_file_exists,
    s3_get_file_list,
    s3_get_file_stream,
    s3_upload_file,
)


@dataclass
class S3OutputDataFolder(BaseOutputDataFolder):
    local_path: str = None
    cleanup: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.path.startswith("s3://"):
            raise ValueError("S3OutputDataFolder path must start with s3://")
        self._tmpdir = None

    def close(self):
        super().close()
        for file in self._output_files.values():
            with self._lock:
                logger.info(f'Uploading "{file.local_path}" to "{file.path}"...')
                s3_upload_file(file.local_path, file.path)
                logger.info(f'Uploaded "{file.local_path}" to "{file.path}".')
                if self.cleanup:
                    os.remove(file.local_path)
        if self.cleanup:
            self._output_files = {}
        if self._tmpdir:
            self._tmpdir.cleanup()

    def create_new_file(self, relative_path: str):
        if not self.local_path:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.local_path = self._tmpdir.name
        return OutputDataFile(
            local_path=os.path.join(self.local_path, relative_path),
            path=os.path.join(self.path, relative_path),
            relative_path=relative_path,
        )


@dataclass
class S3OutputDataFile(OutputDataFile):
    path: str = None
    cleanup: bool = True

    def __post_init__(self):
        if not self.path.startswith("s3://"):
            raise ValueError("S3OutputDataFile path must start with s3://")
        self._tmpdir = None

    def close(self):
        super().close()
        logger.info(f'Uploading "{self.local_path}" to "{self.path}"...')
        s3_upload_file(self.local_path, self.path)
        logger.info(f'Uploaded "{self.local_path}" to "{self.path}".')
        if self.cleanup:
            os.remove(self.local_path)
        if self._tmpdir:
            self._tmpdir.cleanup()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self, mode: str = "w", gzip: bool = False, overwrite: bool = False):
        if not self.local_path:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.local_path = self._tmpdir.name
        if not self.file_handler or overwrite:
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
            self.file_handler = open(self.local_path, mode) if not gzip else gzip_lib.open(self.local_path, mode)
        return self

    def write(self, *args, **kwargs):
        self.file_handler.write(*args, **kwargs)


@dataclass
class S3InputDataFile(InputDataFile):
    local_path: str = None
    stream: bool = False
    folder: "S3InputDataFolder" = None

    @contextmanager
    def open_binary(self):
        if self.stream:
            # stream
            response_stream = s3_get_file_stream(self.path)
            try:
                yield response_stream
            finally:
                response_stream.close()
        else:
            # download
            if not os.path.isfile(self.local_path):
                with self.folder._lock:
                    logger.info(f'Downloading "{self.path}" to "{self.local_path}"...')
                    s3_download_file(self.path, self.local_path)
                    logger.info(f'Downloaded "{self.path}" to "{self.local_path}".')
            with open(self.local_path, mode="rb") as f:
                try:
                    yield f
                finally:
                    if self.folder.cleanup:
                        os.remove(self.local_path)


@dataclass
class S3InputDataFolder(BaseInputDataFolder):
    local_path: str = None
    stream: bool = False
    cleanup: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.path.startswith("s3://"):
            raise ValueError("S3InputDataFolder path must start with s3://")
        self._tmpdir = None

    def unchecked_get_file(self, relative_path: str):
        return S3InputDataFile(
            path=os.path.join(self.path, relative_path),
            local_path=os.path.join(self.local_path, relative_path),
            relative_path=relative_path,
            folder=self,
            stream=self.stream,
        )

    def file_exists(self, relative_path: str) -> bool:
        return s3_file_exists(os.path.join(self.path, relative_path))

    def list_files(self, extension: str | list[str] = None, suffix: str = "") -> list[InputDataFile]:
        if not self.local_path:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.local_path = self._tmpdir.name
        return [
            self.unchecked_get_file(os.path.join(suffix, path))
            for path in s3_get_file_list(
                os.path.join(self.path, suffix), match_pattern=self.match_pattern, recursive=self.recursive
            )
            if self._match_file(path, extension)
        ]
