#!/usr/bin/env python3
"""
Example 1: Basic Data Processing Pipeline

Learn DataTrove fundamentals: read from HuggingFace datasets, apply filters, save results.

Components:
- JsonlReader: Stream from HuggingFace C4 dataset
- LambdaFilter: Custom filtering logic (length, keywords)
- SamplerFilter: Random sampling (50%)
- JsonlWriter: Save filtered results

Usage:
    python spec/phase1/examples/01_basic_filtering.py
    python spec/phase1/examples/01_basic_filtering.py inspect
"""

import json
import os
import sys

from datatrove.utils.logging import logger
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

# Configuration
OUTPUT_DIR = "spec/phase1/output/01_filtered"
LOGS_DIR = "spec/phase1/logs/01_basic_filtering"


def main():
    """Main pipeline execution."""
    logger.info("Starting Example 1: Basic Filtering Pipeline")

    pipeline = [
        JsonlReader(
            "hf://datasets/allenai/c4/en/",
            glob_pattern="c4-train.00000-of-01024.json.gz",
            limit=1000,
        ),
        LambdaFilter(lambda doc: len(doc.text) > 100),
        LambdaFilter(
            lambda doc: any(kw in doc.text.lower()
                          for kw in ["data", "learning", "computer", "science"])
        ),
        SamplerFilter(rate=0.5),
        JsonlWriter(
            output_folder=OUTPUT_DIR,
            output_filename="filtered_${rank}.jsonl",
            compression=None
        )
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir=LOGS_DIR
    )

    executor.run()

    logger.info(f"Pipeline completed. Check: {OUTPUT_DIR}")


def inspect_results():
    """Helper function to inspect the results."""
    output_file = f"{OUTPUT_DIR}/filtered_00000.jsonl"

    if not os.path.exists(output_file):
        logger.warning("No output file found. Run the pipeline first.")
        return

    logger.info("Inspecting Results")

    # Count documents
    with open(output_file, 'r') as f:
        docs = [json.loads(line) for line in f]

    logger.info(f"Total documents after filtering: {len(docs)}")

    if docs:
        # Show sample document
        sample = docs[0]
        logger.info(f"Sample document ID: {sample.get('id', 'N/A')}")
        logger.info(f"Text length: {len(sample.get('text', ''))}")
        logger.info(f"Preview: {sample.get('text', '')[:200]}...")

        # Basic statistics
        text_lengths = [len(doc.get('text', '')) for doc in docs]
        logger.info(f"Text length stats - Min: {min(text_lengths)}, "
                   f"Max: {max(text_lengths)}, "
                   f"Avg: {sum(text_lengths) / len(text_lengths):.0f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        inspect_results()
    else:
        main()
        inspect_results()