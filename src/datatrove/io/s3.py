import os.path
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

from datatrove.io import InputDataFile, LocalOutputDataFolder
from datatrove.io.base import BaseInputDataFolder
from datatrove.io.cloud.s3 import s3_download_file, s3_get_file_list, s3_get_file_stream, s3_upload_file


@dataclass
class S3OutputDataFolder(LocalOutputDataFolder):
    local_path: str = None

    def __post_init__(self):
        super().__post_init__()
        if not self.path.startswith("s3://"):
            raise ValueError("S3OutputDataFolder path must start with s3://")
        self._tmpdir = None

    def close(self, close_fn: Callable = None):
        super().close(close_fn)
        for file in self._output_files.values():
            with self._lock:
                s3_upload_file(file.local_path, file.path)
        if self._tmpdir:
            self._tmpdir.cleanup()

    def create_new_file(self, relative_path: str):
        if not self.local_path:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.local_path = self._tmpdir.name
        return super().create_new_file(relative_path)


@dataclass
class S3InputDataFile(InputDataFile):
    folder: BaseInputDataFolder = None
    stream: bool = False

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
                    s3_download_file(self.path, self.local_path)
            with super().open_binary() as f:
                yield f


@dataclass
class S3InputDataFolder(BaseInputDataFolder):
    local_path: str = None
    stream: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not self.path.startswith("s3://"):
            raise ValueError("S3InputDataFolder path must start with s3://")
        self._tmpdir = None

    def list_files(self, extension: str | list[str] = None) -> list[InputDataFile]:
        if not self.local_path:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.local_path = self._tmpdir.name
        return [
            S3InputDataFile(
                path=os.path.join(self.path, path),
                local_path=os.path.join(self.local_path, path),
                folder=self,
                stream=self.stream,
            )
            for path in s3_get_file_list(self.path, match_pattern=self.match_pattern, recursive=self.recursive)
            if self._match_file(path, extension)
        ]
