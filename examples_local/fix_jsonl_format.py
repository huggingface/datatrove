#!/usr/bin/env python3
"""
Fix JSONL format to be compatible with DataTrove JsonlReader.

The issue: JsonlReader expects Document format, but our data is in CommonCrawl format.
CommonCrawl format has metadata at top level, but JsonlReader's adapter loses it.
"""

import json
import gzip

def fix_jsonl_format():
    """Convert CommonCrawl JSONL format to DataTrove Document format."""

    input_file = "examples_local/test_sample.jsonl.gz"
    output_file = "examples_local/test_sample_fixed.jsonl.gz"

    print("Converting CommonCrawl JSONL format to DataTrove Document format...")

    with gzip.open(input_file, 'rt') as infile, gzip.open(output_file, 'wt') as outfile:
        for line_num, line in enumerate(infile):
            data = json.loads(line)

            print(f"\n--- Processing record {line_num + 1} ---")
            print(f"Original ID: {data['id']}")
            print(f"Original metadata keys: {list(data['metadata'].keys())}")

            # Convert to DataTrove Document format
            # The key insight: move ALL metadata into the document metadata
            # instead of having it split between top-level and nested
            fixed_data = {
                "id": data["id"],
                "text": "",  # Empty text - will be filled by WarcReaderFast
                "media": data["media"],
                "metadata": data["metadata"]  # This preserves warc_filename, warc_record_offset, etc.
            }

            print(f"Fixed metadata keys: {list(fixed_data['metadata'].keys())}")
            print(f"Has warc_filename: {'warc_filename' in fixed_data['metadata']}")
            print(f"Has warc_record_offset: {'warc_record_offset' in fixed_data['metadata']}")

            outfile.write(json.dumps(fixed_data) + '\n')

    print(f"\n✅ Fixed JSONL saved to: {output_file}")

if __name__ == "__main__":
    fix_jsonl_format()