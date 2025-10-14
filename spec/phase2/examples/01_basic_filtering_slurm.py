#!/usr/bin/env python3
"""
Example 01: Basic Filtering with Slurm

Apply basic filtering to C4 data using SlurmPipelineExecutor for distributed processing.

Components:
- JsonlReader: Read from HuggingFace C4 dataset
- LambdaFilter: Filter by length and keywords
- SamplerFilter: Random sampling
- JsonlWriter: Save filtered results
- SlurmPipelineExecutor: Distributed execution across Slurm cluster

Usage:
    python spec/phase2/examples/01_basic_filtering_slurm.py
"""

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.logging import logger

# Configuration
OUTPUT_DIR = "/tmp/output"
LOGS_DIR = "/tmp/logs"
SLURM_LOGS_DIR = "/tmp/slurm_logs"


def main():
    """Main pipeline execution."""
    logger.info("Starting Slurm Basic Filtering Pipeline")

    pipeline = [
        JsonlReader(
            "hf://datasets/allenai/c4/en/",
            glob_pattern="c4-train.00000-of-01024.json.gz",
            limit=100,
        ),
        LambdaFilter(lambda doc: len(doc.text) > 100),
        LambdaFilter(
            lambda doc: any(keyword in doc.text.lower()
                          for keyword in ["data", "learning", "computer", "science"])
        ),
        SamplerFilter(rate=0.5),
        JsonlWriter(OUTPUT_DIR)
    ]

    executor = SlurmPipelineExecutor(
        job_name="basic_filtering",
        pipeline=pipeline,
        tasks=2,
        time="00:05:00",
        partition="gpu",
        logging_dir=LOGS_DIR,
        slurm_logs_folder=SLURM_LOGS_DIR,
        cpus_per_task=8,
        mem_per_cpu_gb=8,
    )

    executor.run()

    logger.info(f"Pipeline submitted. Check Slurm logs: {SLURM_LOGS_DIR}")


if __name__ == "__main__":
    main()