import json
import os
import tempfile

import pytest

from datatrove.pipeline.readers import JsonlReader


# Known bug. strict=True: once fixed this passes and fails CI until the marker is removed.
# Open question: what should happen to the invalid line itself, and to the lines after it?
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="UTF-8 decoder read-ahead discards valid preceding lines"
)
def test_jsonl_reader_keeps_lines_before_an_invalid_byte():
    lines = [json.dumps({"id": str(i), "text": f"text {i}"}).encode() for i in range(5)]
    lines[2] = lines[2].replace(b"text", b"t\xffxt")
    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(folder, "data.jsonl"), "wb") as f:
            f.write(b"\n".join(lines) + b"\n")
        read_ids = [doc.id for doc in JsonlReader(folder)()]
    assert read_ids[:2] == ["0", "1"]
