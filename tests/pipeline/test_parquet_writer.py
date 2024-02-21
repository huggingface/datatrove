import shutil
import tempfile
import unittest

from datatrove.data import Document
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers.parquet import ParquetWriter
from datatrove.utils._import_utils import is_pyarrow_available

from ..utils import require_pyarrow


if is_pyarrow_available():
    pass  # noqa: F811


@require_pyarrow
class TestParquetWriter(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_write(self):
        data = [
            Document(text=text, id=str(i), metadata={"somedata": 2 * i})
            for i, text in enumerate(["hello", "text2", "more text"])
        ]
        with ParquetWriter(output_folder=self.tmp_dir, batch_size=2) as w:
            for doc in data:
                w.write(doc)
        reader = ParquetReader(self.tmp_dir)
        c = 0
        for read_doc, original in zip(reader(), data):
            read_doc.metadata.pop("file_path", None)
            assert read_doc == original
            c += 1
        assert c == len(data)
