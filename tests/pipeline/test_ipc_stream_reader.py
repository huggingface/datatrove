import os
import shutil
import tempfile
import unittest

import pyarrow as pa

from datatrove.io import LocalInputDataFolder
from datatrove.pipeline.readers.ipc_stream import IpcStreamReader


class TestIpcReader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

        # Create a dummy ipc stream file
        self.ipc_stream_file = os.path.join(self.tmp_dir, "data.arrow")
        pa_table = pa.table(
            {"content": ["good", "bad", "equisite"], "data_id": [2, 3, 4], "content_length": [4, 3, 8]}
        )
        with pa.ipc.new_stream(self.ipc_stream_file, pa_table.schema) as writer:
            writer.write_table(pa_table)

    def check_same_data(self, documents, check_metadata=True):
        with pa.ipc.open_stream(self.ipc_stream_file) as ipc_stream_reader:
            rows = ipc_stream_reader.read_all().to_pylist()
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

    def test_ipc_stream_reader(self):
        reader = IpcStreamReader(LocalInputDataFolder(self.tmp_dir))
        documents = list(reader.run())
        self.check_same_data(documents)
