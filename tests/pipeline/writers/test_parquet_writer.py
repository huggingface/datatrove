import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from datatrove.data import Document
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers.parquet import ParquetWriter

from ...utils import require_pyarrow


@require_pyarrow
class TestParquetWriter(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_write(self):
        data = [
            Document(text=text, id=str(i), metadata={"somedata": 2 * i, "somefloat": i * 0.4, "somestring": "hello"})
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

    def test_write_chunked_columns(self) -> None:
        """Write real chunked columns without allocating a multi-GiB string buffer."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        def chunked_table(records: list[dict[str, Any]], schema: Any = None) -> pa.Table:
            """Use a tiny chunk threshold to exercise the wide-column conversion path."""
            table = pa.Table.from_pylist(records, schema=schema)
            cut = len(records) // 2
            return pa.concat_tables([table.slice(0, cut), table.slice(cut)])

        def record_batch(records: list[dict[str, Any]], schema: Any = None) -> pa.RecordBatch:
            """Keep first-document inference, then let Arrow reject the chunked columns."""
            if len(records) == 1:
                return pa.RecordBatch.from_pylist(records, schema=schema)
            table = chunked_table(records, schema=schema)
            return pa.RecordBatch.from_arrays(table.columns, schema=table.schema)

        proxy = types.ModuleType("pyarrow")
        proxy.__dict__.update(pa.__dict__)
        proxy.Table = types.SimpleNamespace(from_pylist=chunked_table)
        proxy.RecordBatch = types.SimpleNamespace(from_pylist=record_batch)
        data = [Document(text=f"Árbol 🌱 {index}", id=str(index), metadata={"seq": index}) for index in range(57)]
        with patch.dict(sys.modules, {"pyarrow": proxy}):
            writer = ParquetWriter(self.tmp_dir, batch_size=25, expand_metadata=True, max_file_size=-1)
            assert list(writer.run(iter(data))) == data

        metadata = pq.read_metadata(Path(self.tmp_dir) / "00000.parquet")
        assert [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)] == [25, 25, 7]
        actual = list(ParquetReader(self.tmp_dir).run())
        for doc in actual:
            doc.metadata.pop("file_path", None)
        assert actual == data

    def test_write_nullable_metadata_and_schema(self) -> None:
        """Preserve nullable metadata in both layouts with inferred or explicit schemas."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        data = [
            Document(
                text=f"Árbol 漢字 🌱 {index}",
                id=str(index),
                metadata={"seq": index, "score": index / 10 if index % 3 == 0 else None},
            )
            for index in range(7)
        ]
        for expanded in [False, True]:
            first = {"text": data[0].text, "id": data[0].id}
            first.update(data[0].metadata if expanded else {"metadata": data[0].metadata})
            for explicit_schema in [False, True]:
                with self.subTest(expanded=expanded, explicit_schema=explicit_schema):
                    folder = Path(self.tmp_dir) / f"expanded-{expanded}-schema-{explicit_schema}"
                    schema = pa.Table.from_pylist([first]).schema if explicit_schema else None
                    writer = ParquetWriter(
                        str(folder), batch_size=3, expand_metadata=expanded, max_file_size=-1, schema=schema
                    )
                    assert list(writer.run(iter(data))) == data
                    actual = list(ParquetReader(str(folder)).run())
                    for doc in actual:
                        doc.metadata.pop("file_path", None)
                    assert actual == data
                    metadata = pq.read_metadata(folder / "00000.parquet")
                    assert [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)] == [
                        3,
                        3,
                        1,
                    ]
