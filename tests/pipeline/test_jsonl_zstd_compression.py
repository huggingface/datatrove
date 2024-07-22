import shutil
import tempfile
import unittest

from datatrove.data import Document
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


class TestZstdCompression(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_jsonl_writer_reader(self):
        data = [
            Document(text=text, id=str(i), metadata={"somedata": 2 * i, "somefloat": i * 0.4, "somestring": "hello"})
            for i, text in enumerate(["hello", "text2", "more text"])
        ]
        with JsonlWriter(output_folder=self.tmp_dir, compression="zstd") as w:
            for doc in data:
                w.write(doc)
        reader = JsonlReader(self.tmp_dir, compression="zstd")
        c = 0
        for read_doc, original in zip(reader(), data):
            read_doc.metadata.pop("file_path", None)
            assert read_doc == original
            c += 1
        assert c == len(data)
