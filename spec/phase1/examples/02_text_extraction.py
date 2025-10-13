#!/usr/bin/env python3
"""
Example 2: Text Extraction and Quality Filtering

Extract clean text from Common Crawl WARC files using Trafilatura and apply quality filters.

Components:
- WarcReader: Read Common Crawl WARC files
- Trafilatura: Extract text from HTML
- LanguageFilter: Keep only English
- GopherRepetitionFilter: Remove repetitive content
- GopherQualityFilter: Apply quality heuristics
- JsonlWriter: Save clean results (+ exclusion writers)

Usage:
    python spec/phase1/examples/02_text_extraction.py
    python spec/phase1/examples/02_text_extraction.py inspect
"""

import glob
import gzip
import json
import os
import sys

from datatrove.utils.logging import logger
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import GopherQualityFilter, GopherRepetitionFilter, LanguageFilter
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

# Configuration
OUTPUT_DIR = "spec/phase1/output"
LOGS_DIR = "spec/phase1/logs/02_text_extraction"


def check_warc_file():
    """Check for existing WARC file."""
    os.makedirs("spec/phase1/data", exist_ok=True)

    sample_file = "spec/phase1/data/CC-MAIN-20180420081400-20180420101400-00000.warc.gz"

    if os.path.exists(sample_file):
        logger.info(f"Found WARC file: {sample_file}")
        try:
            with gzip.open(sample_file, 'rb') as f:
                f.read(1)
            size_mb = os.path.getsize(sample_file) / (1024 * 1024)
            logger.info(f"File size: {size_mb:.1f} MB")
            return sample_file
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return None

    logger.warning(f"WARC file not found at: {sample_file}")
    return None


def main():
    """Main pipeline execution."""
    warc_file = check_warc_file()
    if not warc_file:
        logger.error("Please download a WARC file first!")
        return

    logger.info("Starting Example 2: Text Extraction Pipeline")

    pipeline = [
        WarcReader(
            data_folder="spec/phase1/data",
            glob_pattern="CC-MAIN-20180420081400-20180420101400-00000.warc.gz",
            limit=200,
        ),
        Trafilatura(favour_precision=True),
        LanguageFilter(
            languages=["en"],
            exclusion_writer=JsonlWriter(
                output_folder=f"{OUTPUT_DIR}/02_non_english",
                compression=None
            )
        ),
        GopherRepetitionFilter(
            exclusion_writer=JsonlWriter(
                output_folder=f"{OUTPUT_DIR}/02_repetitive",
                compression=None
            )
        ),
        GopherQualityFilter(
            min_doc_words=50,
            max_doc_words=100000,
            exclusion_writer=JsonlWriter(
                output_folder=f"{OUTPUT_DIR}/02_low_quality",
                compression=None
            )
        ),
        JsonlWriter(
            output_folder=f"{OUTPUT_DIR}/02_clean",
            output_filename="clean_${rank}.jsonl",
            compression=None
        )
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir=LOGS_DIR
    )

    executor.run()

    logger.info("Pipeline completed. Check outputs in spec/phase1/output/02_*/")


def inspect_results():
    """Analyze the filtering results."""
    logger.info("Analyzing Results")

    folders = {
        "Clean": f"{OUTPUT_DIR}/02_clean/*.jsonl",
        "Non-English": f"{OUTPUT_DIR}/02_non_english/*.jsonl",
        "Repetitive": f"{OUTPUT_DIR}/02_repetitive/*.jsonl",
        "Low Quality": f"{OUTPUT_DIR}/02_low_quality/*.jsonl",
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
                        sample = docs[0]
                        logger.info(f"Sample from {category}:")
                        logger.info(f"  URL: {sample.get('metadata', {}).get('url', 'N/A')}")
                        logger.info(f"  Text length: {len(sample.get('text', ''))}")
                        logger.info(f"  Preview: {sample.get('text', '')[:200]}...")

        logger.info(f"{category}: {total_docs} documents")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        inspect_results()
    else:
        main()
        inspect_results()