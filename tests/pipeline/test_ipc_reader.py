import os
import shutil
import tempfile
import unittest

import pyarrow as pa
import pyarrow.feather as feather

from datatrove.io import LocalInputDataFolder
from datatrove.pipeline.readers.ipc import IpcReader


class TestIpcReader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

        # Create a dummy ipc/feather file
        self.ipc_file = os.path.join(self.tmp_dir, "data.feather")
        pa_table = pa.table(
            {"content": ["good", "bad", "equisite"], "data_id": [2, 3, 4], "content_length": [4, 3, 8]}
        )
        feather.write_feather(pa_table, self.ipc_file)

    def check_same_data(self, documents, check_metadata=True):
        rows = feather.read_table(self.ipc_file).to_pylist()
        self.assertEqual(len(documents), len(rows))
        for document, row in zip(documents, rows):
            self.assertEqual(document.content, row["content"])
            data_id = row.get("data_id", None)
            if data_id:
                self.assertEqual(document.data_id, data_id)
            if check_metadata:
                self.assertIsNotNone(document.metadata)
                for key in row.keys() - {"content", "data_id"}:
                    self.assertEqual(document.metadata[key], row[key])

    def test_ipc_reader(self):
        reader = IpcReader(LocalInputDataFolder(self.tmp_dir))
        documents = list(reader.run())
        self.check_same_data(documents)
