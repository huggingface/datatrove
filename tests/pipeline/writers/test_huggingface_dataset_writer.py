"""Tests for HuggingFaceDatasetWriter.

Unit tests mock the underlying ``huggingface_hub`` APIs (``create_repo`` /
``preupload_lfs_files`` / ``create_commit``) so they run without network access.
The fake ``create_commit`` mimics huggingface_hub's real bookkeeping: it flags
each ``CommitOperationAdd`` as committed on success and raises the same
``ValueError`` huggingface_hub raises when a caller tries to reuse an
already-committed operation. This reproduces (and guards against) the writer
reuse bug that occurs when a ``SlurmPipelineExecutor`` with ``tasks_per_job > 1``
runs multiple ranks sequentially against the same writer instance.
"""

import shutil
import tempfile
import unittest
from unittest.mock import patch

from datatrove.data import Document
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers.huggingface import HuggingFaceDatasetWriter

from ...utils import require_pyarrow


def _make_docs(n: int, start: int = 0) -> list[Document]:
    return [Document(text=f"hello {i}", id=f"doc-{i}", metadata={"split": "train"}) for i in range(start, start + n)]


def _fake_create_commit(*_args, operations, **_kwargs):
    for op in operations:
        if getattr(op, "_is_committed", False):
            raise ValueError(
                f"CommitOperationAdd {op} has already being committed and cannot be reused. "
                "Please create a new CommitOperationAdd object if you want to create a new commit."
            )
    for op in operations:
        op._is_committed = True


@require_pyarrow
class TestHuggingFaceDatasetWriterReuse(unittest.TestCase):
    """Regression test for reusing one writer instance across multiple ranks.

    SlurmPipelineExecutor with tasks_per_job > 1 runs several ranks
    sequentially against the same pipeline step instances, so close() must be
    safe to call more than once on the same writer.
    """

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        self._create_repo_patcher = patch("datatrove.pipeline.writers.huggingface.create_repo", return_value=None)
        self._preupload_patcher = patch(
            "datatrove.pipeline.writers.huggingface.preupload_lfs_files", return_value=None
        )
        self._create_commit_patcher = patch(
            "datatrove.pipeline.writers.huggingface.create_commit",
            side_effect=_fake_create_commit,
        )
        self._create_repo_patcher.start()
        self._preupload_patcher.start()
        self.mock_create_commit = self._create_commit_patcher.start()
        self.addCleanup(self._create_repo_patcher.stop)
        self.addCleanup(self._preupload_patcher.stop)
        self.addCleanup(self._create_commit_patcher.stop)

    def _make_writer(self) -> HuggingFaceDatasetWriter:
        return HuggingFaceDatasetWriter(dataset="org/my-dataset", local_working_dir=self.tmp_dir, cleanup=False)

    def test_reused_writer_commits_each_rank_independently(self) -> None:
        writer = self._make_writer()

        with writer:
            writer.write(_make_docs(1)[0], rank=0)
        with writer:
            writer.write(_make_docs(1, start=1)[0], rank=1)

        self.assertEqual(self.mock_create_commit.call_count, 2)
        first_ops = self.mock_create_commit.call_args_list[0].kwargs["operations"]
        second_ops = self.mock_create_commit.call_args_list[1].kwargs["operations"]
        self.assertEqual(len(first_ops), 1)
        self.assertEqual(len(second_ops), 1)
        self.assertNotEqual(first_ops[0], second_ops[0])
        self.assertEqual(writer.operations, [])

    def test_reused_writer_round_trips_all_ranks(self) -> None:
        writer = self._make_writer()
        docs_rank0 = _make_docs(2, start=0)
        docs_rank1 = _make_docs(2, start=2)

        with writer:
            for doc in docs_rank0:
                writer.write(doc, rank=0)
        with writer:
            for doc in docs_rank1:
                writer.write(doc, rank=1)

        reader = ParquetReader(self.tmp_dir)
        read_back = list(reader())
        self.assertEqual(len(read_back), len(docs_rank0) + len(docs_rank1))


if __name__ == "__main__":
    unittest.main()
