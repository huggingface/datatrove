"""Tests for HuggingFaceBucketWriter.

Unit tests use ``unittest.mock.patch`` on the underlying ``huggingface_hub`` APIs
(``create_bucket`` / ``batch_bucket_files``) so they run without network access.
Integration tests write to a local staging directory through the writer and read
the resulting parquet files back via ``ParquetReader``.
"""

import os
import shutil
import tempfile
import unittest
from typing import Any
from unittest.mock import call, patch

from datatrove.data import Document
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers.huggingface import HuggingFaceBucketWriter

from ...utils import require_pyarrow


def _make_docs(n: int) -> list[Document]:
    return [
        Document(
            text=f"hello {i}",
            id=f"doc-{i}",
            metadata={"split": "train", "score": float(i)},
        )
        for i in range(n)
    ]


@require_pyarrow
class TestHuggingFaceBucketWriterUnit(unittest.TestCase):
    """Unit tests with mocked Hub APIs and real local staging."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        self._create_bucket_patcher = patch(
            "datatrove.pipeline.writers.huggingface.create_bucket",
            return_value=None,
        )
        self._batch_patcher = patch(
            "datatrove.pipeline.writers.huggingface.batch_bucket_files",
            return_value=None,
        )
        self.mock_create_bucket = self._create_bucket_patcher.start()
        self.mock_batch_bucket_files = self._batch_patcher.start()
        self.addCleanup(self._create_bucket_patcher.stop)
        self.addCleanup(self._batch_patcher.stop)

    def _make_writer(self, **kwargs: Any) -> HuggingFaceBucketWriter:
        defaults: dict[str, Any] = {
            "bucket": "org/my-bucket",
            "local_working_dir": self.tmp_dir,
        }
        defaults.update(kwargs)
        return HuggingFaceBucketWriter(**defaults)

    def test_init_creates_temp_local_working_dir_when_none(self) -> None:
        writer = HuggingFaceBucketWriter(bucket="org/my-bucket")
        self.assertTrue(writer.local_working_dir.is_local())

    def test_init_rejects_non_local_working_dir(self) -> None:
        with self.assertRaisesRegex(ValueError, "local"):
            HuggingFaceBucketWriter(
                bucket="org/my-bucket",
                local_working_dir="s3://some-bucket/staging/",
            )

    def test_upload_files_creates_bucket_on_first_call_only(self) -> None:
        writer = self._make_writer()
        # create the local files so cleanup does not fail
        for name in ("a.parquet", "b.parquet"):
            with writer.local_working_dir.open(name, "wb") as f:
                f.write(b"x")
        writer.upload_files("a.parquet")
        writer.upload_files("b.parquet")
        self.assertEqual(self.mock_create_bucket.call_count, 1)
        self.mock_create_bucket.assert_called_with("org/my-bucket", private=True, exist_ok=True)

    def test_upload_files_calls_batch_bucket_files_with_resolved_paths(self) -> None:
        writer = self._make_writer(prefix="v1")
        with writer.local_working_dir.open("file.parquet", "wb") as f:
            f.write(b"x")
        writer.upload_files("file.parquet")

        self.assertEqual(self.mock_batch_bucket_files.call_count, 1)
        args, kwargs = self.mock_batch_bucket_files.call_args
        self.assertEqual(args[0], "org/my-bucket")
        add = kwargs["add"]
        self.assertEqual(len(add), 1)
        local_path, remote_path = add[0]
        self.assertEqual(local_path, os.path.join(self.tmp_dir, "file.parquet"))
        self.assertEqual(remote_path, "v1/file.parquet")

    def test_upload_files_with_empty_prefix_has_no_leading_slash(self) -> None:
        writer = self._make_writer(prefix="")
        with writer.local_working_dir.open("file.parquet", "wb") as f:
            f.write(b"x")
        writer.upload_files("file.parquet")

        _, kwargs = self.mock_batch_bucket_files.call_args
        _, remote_path = kwargs["add"][0]
        self.assertEqual(remote_path, "file.parquet")

    def test_upload_files_cleanup_true_removes_local_files(self) -> None:
        writer = self._make_writer(cleanup=True)
        path = os.path.join(self.tmp_dir, "x.parquet")
        with writer.local_working_dir.open("x.parquet", "wb") as f:
            f.write(b"x")
        self.assertTrue(os.path.exists(path))

        writer.upload_files("x.parquet")
        self.assertFalse(os.path.exists(path))

    def test_upload_files_cleanup_false_keeps_local_files(self) -> None:
        writer = self._make_writer(cleanup=False)
        path = os.path.join(self.tmp_dir, "x.parquet")
        with writer.local_working_dir.open("x.parquet", "wb") as f:
            f.write(b"x")

        writer.upload_files("x.parquet")
        self.assertTrue(os.path.exists(path))

    def test_close_uploads_remaining_open_files(self) -> None:
        writer = self._make_writer()
        for doc in _make_docs(3):
            writer.write(doc)
        with patch.object(writer, "upload_files", wraps=writer.upload_files) as mock_upload:
            writer.close()
        # close() must trigger at least one upload of the open file(s).
        self.assertGreaterEqual(mock_upload.call_count, 1)

    def test_close_does_not_create_commit(self) -> None:
        """Buckets are commit-less: close() must not call ``create_commit``."""
        writer = self._make_writer()
        for doc in _make_docs(3):
            writer.write(doc)
        with patch("datatrove.pipeline.writers.huggingface.create_commit") as mock_commit:
            writer.close()
        mock_commit.assert_not_called()

    def test_on_file_switch_uploads_old_file(self) -> None:
        """``_on_file_switch`` (called on rotation) must immediately upload the completed file."""
        writer = self._make_writer()
        with patch.object(writer, "upload_files") as mock_upload:
            writer._on_file_switch("data/00000.parquet", "data/000_00000.parquet", "data/001_00000.parquet")
        mock_upload.assert_called_once_with("data/000_00000.parquet")

    def test_full_write_cycle_triggers_uploads_and_stats(self) -> None:
        writer = self._make_writer()
        docs = _make_docs(5)
        for doc in docs:
            writer.write(doc)
        writer.close()

        # bucket auto-created exactly once.
        self.assertEqual(self.mock_create_bucket.call_count, 1)
        # at least one upload happened.
        self.assertGreaterEqual(self.mock_batch_bucket_files.call_count, 1)
        # stats reflect the writes.
        self.assertEqual(int(writer.stats["total"].total), len(docs))

    # --- overwrite mode ---

    def test_overwrite_deletes_existing_files_on_first_upload(self) -> None:
        """With overwrite=True, existing files at the prefix are deleted before the first upload."""
        fake_existing = [
            _make_fake_bucket_file("v1/data/old_000.parquet"),
            _make_fake_bucket_file("v1/data/old_001.parquet"),
        ]
        with patch(
            "datatrove.pipeline.writers.huggingface.list_bucket_tree",
            return_value=iter(fake_existing),
        ) as mock_list:
            writer = self._make_writer(prefix="v1/data", overwrite=True)
            with writer.local_working_dir.open("file.parquet", "wb") as f:
                f.write(b"x")
            writer.upload_files("file.parquet")

        # list_bucket_tree called once to discover existing files.
        mock_list.assert_called_once_with("org/my-bucket", prefix="v1/data", recursive=True)
        # First batch_bucket_files call deletes the old files.
        delete_call = self.mock_batch_bucket_files.call_args_list[0]
        self.assertEqual(
            delete_call, call("org/my-bucket", delete=["v1/data/old_000.parquet", "v1/data/old_001.parquet"])
        )
        # Second call is the actual upload.
        upload_call = self.mock_batch_bucket_files.call_args_list[1]
        self.assertIn("add", upload_call.kwargs)

    def test_overwrite_deletes_only_once(self) -> None:
        """The delete-before-upload step must happen exactly once, not on every upload_files call."""
        with patch(
            "datatrove.pipeline.writers.huggingface.list_bucket_tree",
            return_value=iter([_make_fake_bucket_file("v1/old.parquet")]),
        ) as mock_list:
            writer = self._make_writer(prefix="v1", overwrite=True)
            for name in ("a.parquet", "b.parquet"):
                with writer.local_working_dir.open(name, "wb") as f:
                    f.write(b"x")
            writer.upload_files("a.parquet")
            writer.upload_files("b.parquet")

        mock_list.assert_called_once()

    def test_overwrite_skips_delete_when_no_existing_files(self) -> None:
        """No delete call when there are no existing files at the prefix."""
        with patch(
            "datatrove.pipeline.writers.huggingface.list_bucket_tree",
            return_value=iter([]),
        ):
            writer = self._make_writer(prefix="v1", overwrite=True)
            with writer.local_working_dir.open("file.parquet", "wb") as f:
                f.write(b"x")
            writer.upload_files("file.parquet")

        # Only one batch call: the upload. No delete call.
        self.assertEqual(self.mock_batch_bucket_files.call_count, 1)
        _, kwargs = self.mock_batch_bucket_files.call_args
        self.assertIn("add", kwargs)

    def test_overwrite_false_by_default(self) -> None:
        """Default behaviour is append — no list/delete happens."""
        writer = self._make_writer()
        with writer.local_working_dir.open("file.parquet", "wb") as f:
            f.write(b"x")
        with patch("datatrove.pipeline.writers.huggingface.list_bucket_tree") as mock_list:
            writer.upload_files("file.parquet")
        mock_list.assert_not_called()


def _make_fake_bucket_file(path: str) -> Any:
    """Return a minimal object that behaves like ``huggingface_hub.BucketFile``."""
    from types import SimpleNamespace

    return SimpleNamespace(type="file", path=path, size=100)


@require_pyarrow
class TestHuggingFaceBucketWriterIntegration(unittest.TestCase):
    """Round-trip integration: write -> stage locally -> read back via ParquetReader."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_parquet_round_trip_via_bucket_writer(self) -> None:
        # cleanup=False so files stay around for ParquetReader to pick up.
        with (
            patch("datatrove.pipeline.writers.huggingface.create_bucket"),
            patch("datatrove.pipeline.writers.huggingface.batch_bucket_files"),
        ):
            writer = HuggingFaceBucketWriter(
                bucket="org/my-bucket",
                local_working_dir=self.tmp_dir,
                cleanup=False,
            )
            originals = _make_docs(7)
            with writer:
                for doc in originals:
                    writer.write(doc)

        reader = ParquetReader(self.tmp_dir)
        read_back = list(reader())
        self.assertEqual(len(read_back), len(originals))
        for read_doc, original in zip(read_back, originals):
            read_doc.metadata.pop("file_path", None)
            self.assertEqual(read_doc.text, original.text)
            self.assertEqual(read_doc.id, original.id)


if __name__ == "__main__":
    unittest.main()
