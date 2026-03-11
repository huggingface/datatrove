import os
import shutil
import tempfile
import unittest

from datatrove.pipeline.readers.csv import CsvReader


class TestCsvReader(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def _write_csv(self, filename: str, content: str) -> str:
        path = os.path.join(self.tmp_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_basic_read(self):
        self._write_csv("data.csv", "text,id\nhello world,doc1\nfoo bar,doc2\n")
        reader = CsvReader(self.tmp_dir, glob_pattern="*.csv")
        docs = list(reader.run(data=None, rank=0, world_size=1))
        assert len(docs) == 2
        assert docs[0].text == "hello world"
        assert docs[0].id == "doc1"

    def test_custom_text_and_id_keys(self):
        self._write_csv("data.csv", "content,uid\nhello,u1\nworld,u2\n")
        reader = CsvReader(self.tmp_dir, text_key="content", id_key="uid", glob_pattern="*.csv")
        docs = list(reader.run(data=None, rank=0, world_size=1))
        assert docs[0].text == "hello"
        assert docs[1].text == "world"

    def test_extra_columns_become_metadata(self):
        self._write_csv("data.csv", "text,id,language,source\nhello,1,en,web\n")
        reader = CsvReader(self.tmp_dir, glob_pattern="*.csv")
        docs = list(reader.run(data=None, rank=0, world_size=1))
        assert docs[0].metadata["language"] == "en"
        assert docs[0].metadata["source"] == "web"

    def test_default_metadata_merged(self):
        self._write_csv("data.csv", "text,id\nhello,1\n")
        reader = CsvReader(self.tmp_dir, default_metadata={"source": "test"}, glob_pattern="*.csv")
        docs = list(reader.run(data=None, rank=0, world_size=1))
        assert docs[0].metadata["source"] == "test"

    def test_limit_and_skip(self):
        rows = "text,id\n" + "\n".join(f"doc{i},{i}" for i in range(10)) + "\n"
        self._write_csv("data.csv", rows)
        reader = CsvReader(self.tmp_dir, skip=2, limit=3, glob_pattern="*.csv")
        docs = list(reader.run(data=None, rank=0, world_size=1))
        assert len(docs) == 3
        assert docs[0].text == "doc2"

    def test_multiple_files(self):
        self._write_csv("a.csv", "text,id\nhello,1\n")
        self._write_csv("b.csv", "text,id\nworld,2\n")
        reader = CsvReader(self.tmp_dir, glob_pattern="*.csv")
        docs = list(reader.run(data=None, rank=0, world_size=1))
        assert len(docs) == 2

    def test_adapter_transforms_data(self):
        self._write_csv("data.csv", "text,id\nhello,1\n")

        def my_adapter(self, data, path, id_in_file):
            return {
                "text": data["text"].upper(),
                "id": data["id"],
                "media": [],
                "metadata": {},
            }

        reader = CsvReader(self.tmp_dir, adapter=my_adapter, glob_pattern="*.csv")
        docs = list(reader.run(data=None, rank=0, world_size=1))
        assert docs[0].text == "HELLO"


if __name__ == "__main__":
    unittest.main()
