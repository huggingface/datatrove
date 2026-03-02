import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from datatrove.data import Document
from datatrove.pipeline.writers.disk_base import DiskWriter


class _TextTestDiskWriter(DiskWriter):
    default_output_filename = "${rank}.txt"

    def _write(self, document: dict[str, Any], file_handler, filename: str) -> None:
        file_handler.write(document["text"])


class _BinaryTestDiskWriter(DiskWriter):
    default_output_filename = "${rank}.txt"

    def __init__(self, output_folder: str, **kwargs: Any) -> None:
        super().__init__(output_folder=output_folder, mode="wb", **kwargs)

    def _write(self, document: dict[str, Any], file_handler, filename: str) -> None:
        file_handler.write(document["text"].encode("utf-8"))


class _DummyFile:
    def __init__(self) -> None:
        self.buffer = ""

    def write(self, text: str) -> None:
        self.buffer += text

    def tell(self) -> int:
        return len(self.buffer)

    def close(self) -> None:
        return None


class _FlakyOutputManager:
    def __init__(self, failures: int, error_message: str) -> None:
        self._failures = failures
        self._error_message = error_message
        self._calls = 0
        self._file = _DummyFile()
        self._open_files = {"00000.txt": self._file}

    @property
    def calls(self) -> int:
        return self._calls

    def get_file(self, filename: str) -> _DummyFile:
        self._calls += 1
        if self._calls <= self._failures:
            raise RuntimeError(self._error_message)
        self._open_files[filename] = self._file
        return self._file

    def get_open_files(self) -> dict[str, _DummyFile]:
        return self._open_files

    def pop(self, filename: str) -> _DummyFile:
        return self._open_files.pop(filename, self._file)

    def close(self) -> None:
        self._open_files.clear()


class _FlakyUploadApi:
    def __init__(self, failures: int, error_message: str) -> None:
        self.failures = failures
        self.error_message = error_message
        self.calls = 0

    def upload_file(self, **_kwargs: Any) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(self.error_message)


class _FailingCloseFile:
    def __init__(self, upload_api: _FlakyUploadApi, close_error_message: str) -> None:
        self.fs = SimpleNamespace(_api=upload_api, token="fake-token")
        self.temp_file = SimpleNamespace(name="/tmp/fake-upload.tmp")
        self.resolved_path = SimpleNamespace(
            path_in_repo="path/in/repo.txt",
            repo_id="org/repo",
            repo_type="dataset",
            revision="main",
        )
        self._close_error_message = close_error_message

    def close(self) -> None:
        raise RuntimeError(self._close_error_message)


class _CloseFileOutputManager:
    def __init__(self, file_obj: _FailingCloseFile) -> None:
        self._file_obj = file_obj

    def get_open_files(self) -> dict[str, _FailingCloseFile]:
        return {}

    def pop(self, _filename: str) -> _FailingCloseFile:
        return self._file_obj

    def close(self) -> None:
        return None


class TestDiskWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_output_filename_uses_rank_and_metadata(self) -> None:
        writer = _TextTestDiskWriter(output_folder=self.tmp_dir, output_filename="${rank}_${id}_${split}.txt")
        doc = Document(text="hello", id="doc-1", metadata={"split": "train"})

        output_filename = writer._get_output_filename(doc, rank=7)

        self.assertEqual(output_filename, "00007_doc-1_train.txt")

    def test_filename_with_file_id_preserves_directory(self) -> None:
        writer = _TextTestDiskWriter(output_folder=self.tmp_dir)

        output_filename = writer._get_filename_with_file_id("subdir/file.txt")

        self.assertEqual(output_filename, "subdir/000_file.txt")

    def test_max_file_size_requires_binary_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_file_size"):
            _TextTestDiskWriter(output_folder=self.tmp_dir, max_file_size=1)

    def test_max_file_size_rotates_output_files(self) -> None:
        with _BinaryTestDiskWriter(output_folder=self.tmp_dir, max_file_size=4) as writer:
            writer.write(Document(text="abcd", id="doc-1"), rank=0)
            writer.write(Document(text="ef", id="doc-2"), rank=0)

        written_files = sorted(os.listdir(self.tmp_dir))
        self.assertEqual(written_files, ["000_00000.txt", "001_00000.txt"])

        with open(os.path.join(self.tmp_dir, "000_00000.txt"), "rb") as file_0:
            self.assertEqual(file_0.read(), b"abcd")
        with open(os.path.join(self.tmp_dir, "001_00000.txt"), "rb") as file_1:
            self.assertEqual(file_1.read(), b"ef")


class TestDiskWriterRetries(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_write_retries_transient_hf_open_errors(self) -> None:
        writer = _TextTestDiskWriter(output_folder=self.tmp_dir)
        writer.output_mg = _FlakyOutputManager(failures=2, error_message="429 Client Error: Too Many Requests")
        doc = Document(text="hello", id="doc-1")

        with patch("datatrove.pipeline.writers.disk_base.time.sleep", return_value=None):
            writer.write(doc, rank=0)

        self.assertEqual(writer.output_mg.calls, 3)

    def test_write_does_not_retry_non_retryable_open_error(self) -> None:
        writer = _TextTestDiskWriter(output_folder=self.tmp_dir)
        writer.output_mg = _FlakyOutputManager(failures=1, error_message="Permission denied")
        doc = Document(text="hello", id="doc-1")

        with patch("datatrove.pipeline.writers.disk_base.time.sleep", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Permission denied"):
                writer.write(doc, rank=0)

    def test_close_file_retries_transient_hf_upload_errors(self) -> None:
        writer = _TextTestDiskWriter(output_folder=self.tmp_dir)
        upload_api = _FlakyUploadApi(failures=2, error_message="A commit has happened since")
        file_obj = _FailingCloseFile(
            upload_api=upload_api,
            close_error_message="A commit has happened since",
        )
        writer.output_mg = _CloseFileOutputManager(file_obj=file_obj)

        with (
            patch("datatrove.pipeline.writers.disk_base.time.sleep", return_value=None),
            patch("datatrove.pipeline.writers.disk_base.os.path.exists", return_value=False),
            patch("datatrove.pipeline.writers.disk_base.os.remove", return_value=None),
        ):
            writer.close_file("00000.txt")

        self.assertEqual(upload_api.calls, 3)
