#!/usr/bin/env python3
"""
Trace exactly what happens in JsonlReader adapter
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
from datatrove.data import Document
from datatrove.pipeline.readers.base import BaseReader

def trace_adapter():
    """Manually trace through the adapter logic."""

    # Load the raw data
    with open("examples_local/test_sample_fixed.jsonl.gz", 'rb') as f:
        import gzip
        with gzip.open(f, 'rt') as gzf:
            raw_line = gzf.readline()
            raw_data = json.loads(raw_line)

    print("=== RAW INPUT DATA ===")
    print(f"Keys: {list(raw_data.keys())}")
    print(f"Metadata keys: {list(raw_data['metadata'].keys())}")
    print(f"warc_filename in metadata: {'warc_filename' in raw_data['metadata']}")

    # Manually run through BaseReader.adapter logic (lines 63-78)
    print("\n=== ADAPTER PROCESSING ===")

    # Line 63: metadata = data.pop("metadata", {})
    data_copy = raw_data.copy()
    metadata = data_copy.pop("metadata", {})
    print(f"1. Popped metadata: {list(metadata.keys())}")
    print(f"   warc_filename in popped metadata: {'warc_filename' in metadata}")

    # Lines 64-72: metadata string parsing (not relevant here)
    print(f"2. Metadata is dict: {isinstance(metadata, dict)}")

    # Line 73-78: return statement
    remaining_data = data_copy.copy()
    text = remaining_data.pop("text", "")
    id_val = remaining_data.pop("id", "test/0")
    media_data = remaining_data.pop("media", [])

    print(f"3. After popping text/id/media, remaining data keys: {list(remaining_data.keys())}")

    # Line 77: FIXED line
    # "metadata": metadata | data
    print(f"4. Using popped metadata: {list(metadata.keys())}")
    print(f"5. remaining data: {remaining_data}")

    final_metadata = metadata | remaining_data
    print(f"6. Final metadata: {list(final_metadata.keys())}")
    print(f"   warc_filename in final: {'warc_filename' in final_metadata}")

    # Create the document
    result = {
        "text": text,
        "id": id_val,
        "media": media_data,
        "metadata": final_metadata
    }

    print(f"\n=== FINAL RESULT ===")
    print(f"Result metadata keys: {list(result['metadata'].keys())}")

    # Test actual Document creation
    doc = Document(**result)
    print(f"Document metadata keys: {list(doc.metadata.keys())}")
    print(f"warc_filename: {doc.metadata.get('warc_filename')}")


if __name__ == "__main__":
    trace_adapter()