#!/usr/bin/env python3
"""
Example 04: Statistics Collection with Slurm

Collect document and language statistics using SlurmPipelineExecutor for distributed processing.

Components:
- JsonlReader: Read multiple C4 files for distribution
- DocStats: Document-level statistics
- LangStats: Language detection statistics
- SlurmPipelineExecutor: Distributed execution with work distribution

Usage:
    python spec/phase2/examples/04_statistics_slurm.py
"""

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.stats import DocStats, LangStats
from datatrove.utils.logging import logger

# Configuration
OUTPUT_DIR = "/tmp/stats"
LOGS_DIR = "/tmp/logs"


def main():
    """Main pipeline execution."""
    logger.info("Starting Slurm Statistics Collection Pipeline")

    pipeline = [
        JsonlReader(
            "hf://datasets/allenai/c4/en/",
            glob_pattern="c4-train.0000[0-3]-of-01024.json.gz",
            limit=200,
        ),
        DocStats(
            output_folder=OUTPUT_DIR,
            histogram_round_digits=1,
        ),
        LangStats(
            output_folder=OUTPUT_DIR,
            language="en",
        ),
    ]

    executor = SlurmPipelineExecutor(
        job_name="stats_collection",
        pipeline=pipeline,
        tasks=2,
        time="00:10:00",
        partition="gpu",
        logging_dir=LOGS_DIR,
        cpus_per_task=8,
        mem_per_cpu_gb=8,
    )

    executor.run()

    logger.info(f"Pipeline submitted. Check stats: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()