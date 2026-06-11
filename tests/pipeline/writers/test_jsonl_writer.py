import json
import os
import shutil
import tempfile
import unittest

from datatrove.data import Document
from datatrove.pipeline.writers.jsonl import JsonlWriter

from ...utils import require_pandas


class TestJsonlWriter(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    @require_pandas
    def test_write_pandas_timestamp_metadata(self):
        # pandas.Timestamp is a datetime.datetime subclass, which orjson refuses to serialize natively
        import pandas as pd

        doc = Document(text="hello", id="0", metadata={"published_date": pd.Timestamp("2021-05-04 22:44:02.776767")})
        with JsonlWriter(output_folder=self.tmp_dir, compression=None) as w:
            w.write(doc)
        with open(os.path.join(self.tmp_dir, "00000.jsonl")) as f:
            written = json.loads(f.readline())
        self.assertEqual(written["metadata"]["published_date"], "2021-05-04T22:44:02.776767")
