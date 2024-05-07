import os
import shutil
import tempfile
import unittest

from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.utils._import_utils import is_pyarrow_available

from ..utils import require_pyarrow


if is_pyarrow_available():
    import pyarrow as pa  # noqa: F811
    import pyarrow.parquet as pq  # noqa: F811


@require_pyarrow
class TestParquetReader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

        # Create a dummy parquet file
        self.parquet_file = os.path.join(self.tmp_dir, "data.parquet")
        pa_table = pa.table({"text": ["good", "bad", "equisite"], "id": [2, 3, 4], "text_length": [4, 3, 8]})
        pq.write_table(pa_table, self.parquet_file)

    def check_same_data(self, documents, check_metadata=True, limit: int | None = None, skip: int = 0):
        rows = pq.read_table(self.parquet_file).to_pylist()
        if skip:
            rows = rows[skip:]
        if limit:
            rows = rows[:limit]

        self.assertEqual(len(documents), len(rows))
        for document, row in zip(documents, rows):
            self.assertEqual(document.text, row["text"])
            id = row.get("id", None)
            if id:
                self.assertEqual(document.id, id)
            if check_metadata:
                self.assertIsNotNone(document.metadata)
                for key in row.keys() - {"text", "id"}:
                    self.assertEqual(document.metadata[key], row[key])

    def test_read(self):
        reader = ParquetReader(self.tmp_dir)
        documents = list(reader.run())
        self.check_same_data(documents)

    def test_read_no_metadata(self):
        reader = ParquetReader(self.tmp_dir, read_metadata=False)
        documents = list(reader.run())
        self.check_same_data(documents, check_metadata=False)

    def test_read_limit(self):
        reader = ParquetReader(self.tmp_dir, limit=1)
        documents = list(reader.run())
        self.assertEqual(len(documents), 1)
        self.check_same_data(documents, limit=1)

    def test_read_skip(self):
        reader = ParquetReader(self.tmp_dir, skip=1)
        documents = list(reader.run())
        self.assertEqual(len(documents), 2)
        self.check_same_data(documents, skip=1)

    def test_read_limit_skip(self):
        reader = ParquetReader(self.tmp_dir, limit=1, skip=1)
        documents = list(reader.run())
        self.assertEqual(len(documents), 1)
        self.check_same_data(documents, limit=1, skip=1)
