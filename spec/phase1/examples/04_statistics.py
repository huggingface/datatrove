#!/usr/bin/env python3
"""
Example 4: Statistics Collection

Collect and analyze document, word, line, and language statistics from processed data.

Components:
- JsonlReader: Read tokenized documents from Example 3
- DocStats: Document-level statistics
- WordStats: Word-level analysis
- LineStats: Line structure metrics
- LangStats: Language detection statistics

Usage:
    python spec/phase1/examples/04_statistics.py
    python spec/phase1/examples/04_statistics.py analyze
"""

import json
import os
import shutil
import sys
from pathlib import Path

from datatrove.utils.logging import logger
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.stats import DocStats, WordStats, LineStats, LangStats, TopKConfig

# Configuration
OUTPUT_DIR = "spec/phase1/output/04_stats"
LOGS_DIR = "spec/phase1/logs/04_statistics"


def main():
    """Main pipeline execution."""
    logger.info("Starting Example 4: Statistics Collection")

    pipeline = [
        JsonlReader(
            data_folder="spec/phase1/output/03_tokenized",
            glob_pattern="*.jsonl",
        ),
        DocStats(
            output_folder=OUTPUT_DIR,
            histogram_round_digits=1,
        ),
        WordStats(
            output_folder=OUTPUT_DIR,
            histogram_round_digits=0,
            top_k_config=TopKConfig(top_k_groups=["fqdn", "suffix"], top_k=100),
        ),
        LineStats(
            output_folder=OUTPUT_DIR,
            histogram_round_digits=0,
        ),
        LangStats(
            output_folder=OUTPUT_DIR,
            language="en",
        ),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir=LOGS_DIR
    )

    executor.run()

    logger.info(f"Statistics collection completed. Stats saved in: {OUTPUT_DIR}")


def analyze_statistics():
    """Analyze the collected statistics."""
    stats_dir = Path(OUTPUT_DIR)

    if not stats_dir.exists():
        logger.warning("No stats found. Run the pipeline first.")
        return

    logger.info("Statistics Analysis")

    # Document length statistics
    doc_length_file = stats_dir / "summary/length/00000.json"
    if doc_length_file.exists():
        with open(doc_length_file, 'r') as f:
            stats = json.load(f)["summary"]
            logger.info(f"Document Statistics - Total chars: {stats['total']:,}, "
                       f"Docs: {stats['n']}, Avg: {stats['mean']:.0f} chars, "
                       f"Min: {stats['min']}, Max: {stats['max']:,}")

    # Whitespace ratio
    ws_file = stats_dir / "summary/white_space_ratio/00000.json"
    if ws_file.exists():
        with open(ws_file, 'r') as f:
            stats = json.load(f)["summary"]
            logger.info(f"Whitespace ratio: {stats['mean']:.2%}")

    # Line statistics
    lines_file = stats_dir / "summary/n_lines/00000.json"
    if lines_file.exists():
        with open(lines_file, 'r') as f:
            stats = json.load(f)["summary"]
            logger.info(f"Line Statistics - Total: {stats['total']:,}, "
                       f"Avg per doc: {stats['mean']:.1f}, "
                       f"Min: {stats['min']}, Max: {stats['max']}")

    # Language statistics
    lang_file = stats_dir / "summary/fasttext_en/00000.json"
    if lang_file.exists():
        with open(lang_file, 'r') as f:
            stats = json.load(f)["summary"]
            logger.info(f"Language Statistics - English confidence (0-1): "
                       f"Avg: {stats['mean']:.3f}, "
                       f"Min: {stats['min']:.3f}, Max: {stats['max']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_statistics()
    else:
        # Clear previous completions to force re-run
        completions_dir = f"{LOGS_DIR}/completions"
        if os.path.exists(completions_dir):
            shutil.rmtree(completions_dir)

        main()
        analyze_statistics()