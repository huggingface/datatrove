import json
import tempfile
from datetime import datetime, timezone

import numpy as np
import pytest


pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

from datatrove.data import Document
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

from ...strategies import doc_text


json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**63), max_value=2**63 - 1)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=10), children, max_size=3),
    max_leaves=10,
)
# include keys that the reader or writer also use, to check they do not clash with user metadata
metadata_keys = st.one_of(st.sampled_from(["file_path", "id", "text", "metadata"]), st.text(max_size=10))


# The reader skips documents with empty text, and the writer drops an empty id, both by design:
# texts are non-empty and ids are set explicitly.
def write_and_read(docs: list[Document]) -> list[Document]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        with JsonlWriter(output_folder=tmp_dir, compression=None) as writer:
            for doc in docs:
                writer.write(doc)
        return list(JsonlReader(tmp_dir, add_file_path=False)())


@given(
    texts=st.lists(doc_text(allow_empty=False), min_size=1, max_size=5),
    metadata=st.dictionaries(metadata_keys, json_values, max_size=4),
)
def test_jsonl_round_trip(texts, metadata):
    docs = [Document(text=text, id=str(i), metadata=dict(metadata)) for i, text in enumerate(texts)]
    read_docs = write_and_read(docs)

    assert [(doc.id, doc.text) for doc in read_docs] == [(doc.id, doc.text) for doc in docs]
    for original, read_doc in zip(docs, read_docs):
        # compare serialized forms: plain == treats True == 1 and 0.0 == -0.0
        assert json.dumps(read_doc.metadata, sort_keys=True) == json.dumps(original.metadata, sort_keys=True)


# Date-like metadata, e.g. timestamp columns read with ParquetReader, is written as ISO strings; missing values as null.
def date_values():
    shapes = [
        st.dates().map(lambda d: (d, d.isoformat())),
        st.datetimes().map(lambda dt: (dt, dt.isoformat())),
        st.datetimes(timezones=st.just(timezone.utc)).map(lambda dt: (dt, dt.isoformat())),
    ]
    try:
        import pandas as pd
    except ImportError:  # pandas is optional: test plain datetimes only
        return st.one_of(shapes)
    # pandas.Timestamp covers roughly 1677-2262
    pandas_range = st.datetimes(min_value=datetime(1678, 1, 1), max_value=datetime(2261, 12, 31))
    shapes.append(pandas_range.map(lambda dt: (pd.Timestamp(dt), pd.Timestamp(dt).isoformat())))
    shapes.append(st.just((pd.NaT, None)))
    return st.one_of(shapes)


@given(values=st.lists(date_values(), min_size=1, max_size=4))
def test_jsonl_writes_dates_as_iso_strings(values):
    metadata = {f"key_{i}": value for i, (value, _) in enumerate(values)}
    (read_doc,) = write_and_read([Document(text="text", id="0", metadata=metadata)])
    assert read_doc.metadata == {f"key_{i}": expected for i, (_, expected) in enumerate(values)}


# Metadata often holds numpy scalars, e.g. a score from a model or an array operation.
@pytest.mark.parametrize("value", [np.int64(1), np.int32(-3), np.float64(0.5), np.float32(0.25), np.bool_(True)])
def test_jsonl_writes_numpy_scalars(value):
    (read_doc,) = write_and_read([Document(text="text", id="0", metadata={"value": value})])
    assert read_doc.metadata == {"value": value.item()}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.float32(0.1), 0.1),  # written with float32 precision, not as 0.10000000149011612
        (np.array([1, 2, 3]), [1, 2, 3]),
        (np.datetime64("2020-01-02"), "2020-01-02T00:00:00"),
    ],
)
def test_jsonl_writes_numpy_values(value, expected):
    (read_doc,) = write_and_read([Document(text="text", id="0", metadata={"value": value})])
    assert read_doc.metadata == {"value": expected}
