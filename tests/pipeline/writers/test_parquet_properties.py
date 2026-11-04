import json
import tempfile

import pytest


pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

from datatrove.data import Document
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from tests.utils import require_pyarrow

from ...strategies import doc_text


# Oracle: documents written with ParquetWriter and read back with ParquetReader keep their text, id and metadata,
# across batch boundaries. Parquet needs one type per column, so each metadata key gets one scalar type, with
# different values per document. Keys that clash with document fields are left out: with expand_metadata=True
# they are not supported.
# "rank" is also excluded: metadata values fill output filename placeholders, so a "rank" key changes the file
RESERVED_KEYS = {"text", "id", "media", "metadata", "file_path", "rank"}
SCALAR_TYPES = [
    st.integers(min_value=-(2**63), max_value=2**63 - 1),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.text(max_size=20),
]
metadata_keys = st.text(min_size=1, max_size=10).filter(lambda key: key not in RESERVED_KEYS)


@st.composite
def documents(draw):
    keys = draw(st.lists(metadata_keys, unique=True, max_size=4))
    key_types = [draw(st.sampled_from(SCALAR_TYPES)) for _ in keys]
    n_docs = draw(st.integers(1, 12))
    return [
        Document(
            text=draw(doc_text(allow_empty=False)),
            id=str(i),
            metadata={key: draw(key_type) for key, key_type in zip(keys, key_types)},
        )
        for i in range(n_docs)
    ]


@require_pyarrow
@given(docs=documents(), batch_size=st.integers(1, 4), expand_metadata=st.booleans())
def test_parquet_round_trip(docs, batch_size, expand_metadata):
    with tempfile.TemporaryDirectory() as folder:
        with ParquetWriter(output_folder=folder, batch_size=batch_size, expand_metadata=expand_metadata) as writer:
            for doc in docs:
                writer.write(doc)
        read_docs = list(ParquetReader(folder)())

    assert [(doc.id, doc.text) for doc in read_docs] == [(doc.id, doc.text) for doc in docs]
    for original, read_doc in zip(docs, read_docs):
        read_doc.metadata.pop("file_path", None)  # added by the reader by default
        # compare serialized forms: plain == treats True == 1 and 0.0 == -0.0
        assert json.dumps(read_doc.metadata, sort_keys=True) == json.dumps(original.metadata, sort_keys=True)
