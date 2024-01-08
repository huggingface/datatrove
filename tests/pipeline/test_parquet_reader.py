import os
import shutil
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from datatrove.io import LocalInputDataFolder
from datatrove.pipeline.readers.parquet import ParquetReader


class TestParquetReader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

        # Create a dummy parquet file
        self.parquet_file = os.path.join(self.tmp_dir, "data.parquet")
        pa_table = pa.table({"text": ["good", "bad", "equisite"], "data_id": [2, 3, 4], "text_length": [4, 3, 8]})
        pq.write_table(pa_table, self.parquet_file)

    def check_same_data(self, documents, check_metadata=True):
        rows = pq.read_table(self.parquet_file).to_pylist()
        self.assertEqual(len(documents), len(rows))
        for document, row in zip(documents, rows):
            self.assertEqual(document.text, row["text"])
            data_id = row.get("data_id", None)
            if data_id:
                self.assertEqual(document.data_id, data_id)
            if check_metadata:
                self.assertIsNotNone(document.metadata)
                for key in row.keys() - {"text", "data_id"}:
                    self.assertEqual(document.metadata[key], row[key])

    def test_read(self):
        reader = ParquetReader(LocalInputDataFolder(self.tmp_dir))
        documents = list(reader.run())
        self.check_same_data(documents)

    def test_read_no_metadata(self):
        reader = ParquetReader(LocalInputDataFolder(self.tmp_dir), read_metadata=False)
        documents = list(reader.run())
        self.check_same_data(documents, check_metadata=False)
