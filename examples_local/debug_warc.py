#!/usr/bin/env python3
"""
Debug WarcReaderFast metadata issue
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
from datatrove.data import Document
from datatrove.pipeline.readers import JsonlReader

def debug_warc_metadata():
    """Debug what metadata the Document objects actually have."""

    # Read the FIXED test file
    test_file = "examples_local/test_sample_fixed.jsonl.gz"

    reader = JsonlReader(
        data_folder="examples_local",
        glob_pattern="test_sample_fixed.jsonl.gz",
        limit=1
    )

    print("Reading documents from JsonlReader...")

    for doc in reader.run():
        print(f"\n--- Document {doc.id} ---")
        print(f"Document type: {type(doc)}")
        print(f"Document.metadata type: {type(doc.metadata)}")
        print(f"Document.metadata keys: {list(doc.metadata.keys())}")

        print(f"\nChecking warc_filename:")
        warc_filename = doc.metadata.get("warc_filename")
        print(f"  Value: {warc_filename}")
        print(f"  Type: {type(warc_filename)}")
        print(f"  Bool: {bool(warc_filename)}")

        print(f"\nChecking warc_record_offset:")
        warc_offset = doc.metadata.get("warc_record_offset")
        print(f"  Value: {warc_offset}")
        print(f"  Type: {type(warc_offset)}")
        print(f"  is not None: {warc_offset is not None}")

        print(f"\nOverall condition:")
        condition = bool(warc_filename) and (warc_offset is not None)
        print(f"  warc_filename and warc_record_offset is not None: {condition}")

        break  # Only check first record


if __name__ == "__main__":
    debug_warc_metadata()