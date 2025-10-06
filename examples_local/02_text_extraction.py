"""
Example 2: Text Extraction and Quality Filtering
=================================================
Extract text from Common Crawl WARC files using Trafilatura,
then apply language and quality filters.

This example demonstrates:
- Reading WARC files from Common Crawl
- Extracting clean text from HTML using Trafilatura
- Language detection and filtering
- Quality filtering with Gopher filters
"""

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
import os


def check_warc_file():
    """Check for existing WARC file"""

    # Create data directory if it doesn't exist
    os.makedirs("examples_local/data", exist_ok=True)

    # The actual file downloaded
    sample_file = "examples_local/data/CC-MAIN-20180420081400-20180420101400-00000.warc.gz"

    if os.path.exists(sample_file):
        print(f"Found WARC file: {sample_file}")
        # Check if it's actually a valid gzip file
        try:
            import gzip
            with gzip.open(sample_file, 'rb') as f:
                f.read(1)
            # Get file size
            size_mb = os.path.getsize(sample_file) / (1024 * 1024)
            print(f"File size: {size_mb:.1f} MB")
            return sample_file
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    print(f"WARC file not found at: {sample_file}")
    return None


def main():
    # Check for WARC file
    warc_file = check_warc_file()
    if not warc_file:
        print("Please download a WARC file first!")
        return

    # Create the processing pipeline
    pipeline = [
        # Read WARC file
        WarcReader(
            data_folder="examples_local/data",
            glob_pattern="CC-MAIN-20180420081400-20180420101400-00000.warc.gz",
            limit=200,  # Process 200 records to get some through filters
        ),

        # Extract text from HTML
        Trafilatura(
            favour_precision=True,  # Prefer precision over recall
        ),

        # Filter by language (keep only English)
        LanguageFilter(
            languages=["en"],
            exclusion_writer=JsonlWriter(
                output_folder="examples_local/output/02_non_english",
                compression=None
            )
        ),

        # Filter out repetitive content
        GopherRepetitionFilter(
            exclusion_writer=JsonlWriter(
                output_folder="examples_local/output/02_repetitive",
                compression=None
            )
        ),

        # Apply quality filter
        GopherQualityFilter(
            min_doc_words=50,
            max_doc_words=100000,
            exclusion_writer=JsonlWriter(
                output_folder="examples_local/output/02_low_quality",
                compression=None
            )
        ),

        # Save the clean results
        JsonlWriter(
            output_folder="examples_local/output/02_clean",
            output_filename="clean_${rank}.jsonl",
            compression=None
        )
    ]

    print("Starting Example 2: Text Extraction Pipeline")
    print("=" * 50)

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir="examples_local/logs/02_text_extraction"
    )

    executor.run()

    print("\n" + "=" * 50)
    print("Pipeline completed!")
    print("Check outputs in:")
    print("  - Clean: examples_local/output/02_clean/")
    print("  - Non-English: examples_local/output/02_non_english/")
    print("  - Repetitive: examples_local/output/02_repetitive/")
    print("  - Low Quality: examples_local/output/02_low_quality/")


def inspect_results():
    """Analyze the filtering results"""
    import json
    import glob

    print("\n" + "=" * 50)
    print("Analyzing Results")
    print("=" * 50)

    folders = {
        "Clean": "examples_local/output/02_clean/*.jsonl",
        "Non-English": "examples_local/output/02_non_english/*.jsonl",
        "Repetitive": "examples_local/output/02_repetitive/*.jsonl",
        "Low Quality": "examples_local/output/02_low_quality/*.jsonl",
    }

    for category, pattern in folders.items():
        files = glob.glob(pattern)
        total_docs = 0

        for file in files:
            if os.path.exists(file):
                with open(file, 'r') as f:
                    docs = [json.loads(line) for line in f if line.strip()]
                    total_docs += len(docs)

                    if docs and category == "Clean":
                        # Show a sample from clean documents
                        print(f"\nSample from {category}:")
                        sample = docs[0]
                        print(f"  URL: {sample.get('metadata', {}).get('url', 'N/A')}")
                        print(f"  Text length: {len(sample.get('text', ''))}")
                        print(f"  Preview: {sample.get('text', '')[:200]}...")

        print(f"\n{category}: {total_docs} documents")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        inspect_results()
    else:
        main()
        inspect_results()