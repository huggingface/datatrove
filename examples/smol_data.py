"""Smol data: Create ~100B token subsets, 50-30-20 mixtures, and shuffled versions.

This script created this dataset: https://huggingface.co/collections/HuggingFaceFW/smol-data

Creates:
  - Individual ~100B subsets: dclm_100BT, fineweb_100BT, fineweb_edu_100BT, finepdfs_100BT, finepdfs_edu_100BT
  - Mix 1 (100B): finepdfs_50BT-dclm_30BT-fineweb_edu_20BT
  - Mix 2 (100B): finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT
  - Shuffled versions of all above (with -shuffled suffix)

Usage:
  python examples/smol_data.py subsets    # Step 1: Create individual ~100B subsets
  python examples/smol_data.py mixtures   # Step 2: Create mixtures (run after Step 1 completes)
  python examples/smol_data.py shuffle    # Step 3: Shuffle all datasets (run after Steps 1+2 complete)
"""

import logging
import os
import subprocess
import sys

import pyarrow as pa

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers.huggingface import HuggingFaceDatasetWriter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration (adjust as needed) ---
SEED = 42
ORG = "HuggingFaceFW"
BASE = "smol_data"
PARTITION = "hopper-cpu"
SHUFFLE_PARTITION = "hopper-prod"
TARGET = 100_000_000_000  # 100B tokens
NUM_TASKS = 100

# Explicit parquet schema for FineWeb-edu (and FineWeb).
# Prevents PyArrow from inferring `null` type when the first document has missing fields (e.g. date).
FINEWEB_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("id", pa.string()),
        ("dump", pa.string()),
        ("url", pa.string()),
        ("date", pa.string()),
        ("file_path", pa.string()),
        ("language", pa.string()),
        ("language_score", pa.float64()),
        ("token_count", pa.int64()),
        ("score", pa.float64()),
        ("int_score", pa.int64()),
        ("dataset", pa.string()),
    ]
)

# Intersection of columns across dclm, fineweb, and finepdfs for mixture datasets.
# dclm:    text, id, url, language, language_score, fasttext_score, dataset
# fineweb: text, id, dump, url, date, file_path, language, language_score, token_count, score, int_score, dataset
# finepdfs: text, id, dump, url, date, file_path, language, token_count, ..., dataset
MIXTURE_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("id", pa.string()),
        ("url", pa.string()),
        ("dataset", pa.string()),
    ]
)

# Source datasets: name -> (hf_id, dataset_options, estimated_total_tokens, schema)
# Token estimates are approximate; SamplerFilter rate = TARGET / total * 1.05
# Measured with estimate_tokens.py (gemma-3-1b-it tokenizer)
SOURCES = {
    "dclm_100BT": (
        "mlfoundations/dclm-baseline-1.0-parquet",
        {"split": "train"},
        3_468_923_154_406,  # 3.5T (1270 tok/doc × 2,732,074,726 docs)
        None,
    ),
    "fineweb_100BT": (
        "HuggingFaceFW/fineweb",
        {"split": "train"},
        16_909_638_933_587,  # 16,910B (653 tok/doc × 25,886,364,489 docs)
        FINEWEB_SCHEMA,
    ),
    "fineweb_edu_100BT": (
        "HuggingFaceFW/fineweb-edu",
        {"split": "train"},
        1_567_210_463_942,  # 1,567B (1028 tok/doc × 1,525,223,056 docs)
        FINEWEB_SCHEMA,
    ),
    "finepdfs_100BT": (
        "HuggingFaceFW/finepdfs",
        {"name": "eng_Latn", "split": "train"},
        726_126_526_327,  # 726B (3509 tok/doc × 206,917,202 docs)
        None,
    ),
    "finepdfs_edu_100BT": (
        "HuggingFaceFW/finepdfs-edu",
        {"name": "eng_Latn", "split": "train"},
        135_905_064_391,  # 136B (5903 tok/doc × 23,023,372 docs)
        None,
    ),
}

# Mixtures: name -> [(source_subset_name, fraction_of_100BT)]
MIXTURES = {
    "finepdfs_50BT-dclm_30BT-fineweb_edu_20BT": [
        ("finepdfs_100BT", 0.5),
        ("dclm_100BT", 0.3),
        ("fineweb_edu_100BT", 0.2),
    ],
    "finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT": [
        ("finepdfs_edu_100BT", 0.5),
        ("dclm_100BT", 0.3),
        ("fineweb_edu_100BT", 0.2),
    ],
}


def _slurm(name: str, pipeline: list, tasks: int, time: str, depends=None) -> SlurmPipelineExecutor:
    """Helper to create a SlurmPipelineExecutor with shared defaults."""
    return SlurmPipelineExecutor(
        job_name=f"nc-{name}",
        pipeline=pipeline,
        tasks=tasks,
        time=time,
        partition=PARTITION,
        logging_dir=f"{BASE}/logs/{name}",
        mem_per_cpu_gb=16,
        cpus_per_task=2,
        depends=depends,
    )


def create_subsets() -> None:
    """Step 1: Create ~100B token subsets for each source dataset."""
    for name, (hf_id, opts, total_tokens, schema) in SOURCES.items():
        # Slightly oversample (1.05x) to ensure we reach ~100B despite variance
        rate = min(1.0, TARGET / total_tokens * 1.05)
        logger.info(f"Submitting {name}: rate={rate:.4f}, tasks={NUM_TASKS}")

        _slurm(
            name=name,
            pipeline=[
                HuggingFaceDatasetReader(hf_id, dataset_options=opts, streaming=True),
                SamplerFilter(rate=rate, seed=SEED),
                HuggingFaceDatasetWriter(
                    dataset=f"{ORG}/{name}",
                    private=False,
                    local_working_dir=f"{BASE}/working/{name}",
                    cleanup=True,
                    schema=schema,
                ),
            ],
            tasks=NUM_TASKS,
            time="3-00:00:00",
        ).run()


def create_mixtures() -> None:
    """Step 2: Create 50-30-20 mixtures by subsampling from the 100B subsets.

    Components for each mix are chained sequentially via Slurm dependencies
    to avoid concurrent commits to the same HF repo.
    """
    for mix_name, components in MIXTURES.items():
        prev = None
        for src, frac in components:
            logger.info(f"Submitting {mix_name}/{src}: frac={frac}")
            prev = _slurm(
                name=f"{mix_name}-{src}",
                pipeline=[
                    HuggingFaceDatasetReader(
                        f"{ORG}/{src}",
                        dataset_options={"split": "train"},
                        streaming=True,
                    ),
                    SamplerFilter(rate=frac, seed=SEED),
                    HuggingFaceDatasetWriter(
                        dataset=f"{ORG}/{mix_name}",
                        private=False,
                        local_working_dir=f"{BASE}/working/{mix_name}/{src}",
                        output_filename=f"{src}/${{rank}}.parquet",
                        cleanup=True,
                        schema=MIXTURE_SCHEMA,
                    ),
                ],
                tasks=NUM_TASKS,
                time="12:00:00",
                depends=prev,
            )
        # Launching the last executor triggers the full chain via dependencies
        prev.run()


def shuffle_datasets() -> None:
    """Step 3: Shuffle all subset and mixture datasets in memory and re-upload.

    Submits one Slurm job per dataset to a GPU node (for large RAM).
    Each job loads the dataset, shuffles it, and pushes as {name}-shuffled.
    """
    # Collect all dataset names to shuffle: subsets + mixtures
    dataset_ids = [f"{ORG}/{name}" for name in SOURCES]
    dataset_ids += [f"{ORG}/{mix_name}" for mix_name in MIXTURES]

    log_dir = f"{BASE}/logs/shuffle"
    os.makedirs(log_dir, exist_ok=True)
    for dataset_id in dataset_ids:
        output_id = f"{dataset_id}-shuffled"
        # Python script executed inside the Slurm job
        # Use num_shards=100 to keep file count under HF Hub commit limit (~130)
        # and individual file sizes under the 5GB LFS limit
        script = (
            "from datasets import load_dataset; "
            f"ds = load_dataset('{dataset_id}', split='train'); "
            f"ds = ds.shuffle(seed={SEED}); "
            f"ds.push_to_hub('{output_id}', private=False, num_shards=100, num_proc=16)"
        )
        cmd = [
            "sbatch",
            f"--job-name=nc-shuffle-{dataset_id.split('/')[-1]}",
            f"--partition={SHUFFLE_PARTITION}",
            "--nodes=1",
            "--ntasks=1",
            "--gpus=1",
            "--cpus-per-task=88",
            "--mem=0",  # Use all available memory on the node
            "--time=24:00:00",
            f"--output={log_dir}/{dataset_id.split('/')[-1]}-%j.out",
            f'--wrap=python -c "{script}"',
        ]
        logger.info(f"Submitting shuffle job for {dataset_id} -> {output_id}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed for {dataset_id}: {result.stderr}")
        logger.info(f"  {result.stdout.strip()}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "subsets"
    if step == "subsets":
        create_subsets()
    elif step == "mixtures":
        create_mixtures()
    elif step == "shuffle":
        shuffle_datasets()
    else:
        logger.error(f"Unknown step: {step}. Use 'subsets', 'mixtures', or 'shuffle'.")
        sys.exit(1)
