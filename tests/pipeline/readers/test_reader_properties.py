import json
import os
import tempfile

import pytest


pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

from datatrove.pipeline.readers import JsonlReader, ParquetReader
from tests.utils import require_pyarrow


# Oracle: across all ranks, a sharded read returns every document exactly once, and skip/limit on one rank
# return the matching slice of that rank's full stream.


def make_rows(file_sizes):
    return [[{"text": f"text {f} {i}", "id": f"{f}_{i}"} for i in range(size)] for f, size in enumerate(file_sizes)]


def write_jsonl_files(folder, files):
    for file_index, rows in enumerate(files):
        with open(os.path.join(folder, f"{file_index:03d}.jsonl"), "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)


def write_parquet_files(folder, files):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("text", pa.string()), ("id", pa.string())])
    for file_index, rows in enumerate(files):
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), os.path.join(folder, f"{file_index:03d}.parquet"))


def assert_sharded_read_is_exact(make_reader, files, world_size, skip, limit):
    read_ids = []
    for rank in range(world_size):
        rank_ids = [doc.id for doc in make_reader()(rank=rank, world_size=world_size)]
        read_ids += rank_ids
        sliced = [doc.id for doc in make_reader(skip=skip, limit=limit)(rank=rank, world_size=world_size)]
        assert sliced == (rank_ids[skip:] if limit == -1 else rank_ids[skip : skip + limit])
    assert sorted(read_ids) == sorted(row["id"] for rows in files for row in rows)


shard_params = {
    "file_sizes": st.lists(st.integers(0, 6), min_size=1, max_size=6),
    "world_size": st.integers(1, 8),
    "skip": st.integers(0, 8),
    "limit": st.integers(-1, 8),
}


@given(**shard_params)
def test_sharded_jsonl_read_returns_every_document_once(file_sizes, world_size, skip, limit):
    files = make_rows(file_sizes)
    with tempfile.TemporaryDirectory() as folder:
        write_jsonl_files(folder, files)
        assert_sharded_read_is_exact(lambda **kwargs: JsonlReader(folder, **kwargs), files, world_size, skip, limit)


@require_pyarrow
@given(**shard_params)
def test_sharded_parquet_read_returns_every_document_once(file_sizes, world_size, skip, limit):
    files = make_rows(file_sizes)
    with tempfile.TemporaryDirectory() as folder:
        write_parquet_files(folder, files)
        reader = lambda **kwargs: ParquetReader(folder, batch_size=2, **kwargs)  # noqa: E731
        assert_sharded_read_is_exact(reader, files, world_size, skip, limit)
