"""Live Slurm test: write parquet shards to a bucket, then read them back.

Run manually from a login node (do **not** submit from inside a Slurm task):

    python tests/manual/test_bucket_slurm_pipeline.py \
        --bucket myorg/test-datatrove-slurm \
        --partition <slurm-partition>

Verifies that ``SlurmPipelineExecutor`` can write parquet shards to a bucket
via ``ParquetWriter(output_folder="hf://buckets/...")`` and that all shards
appear under the expected prefix.
"""

from __future__ import annotations

import argparse
import getpass
import sys
import uuid

from huggingface_hub import HfFileSystem, delete_bucket

from datatrove.data import Document
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers.parquet import ParquetWriter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default=f"{getpass.getuser()}/test-datatrove-{uuid.uuid4().hex[:8]}",
    )
    parser.add_argument("--partition", required=True)
    parser.add_argument("--qos", default="normal")
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n_docs_per_task", type=int, default=10)
    args = parser.parse_args()

    docs = [Document(text=f"hello {i}", id=f"doc-{i}") for i in range(args.tasks * args.n_docs_per_task)]
    output_url = f"hf://buckets/{args.bucket}/v1/parquet"
    logs_url = f"hf://buckets/{args.bucket}/logs/v1"

    print(f"Submitting Slurm job (tasks={args.tasks}, workers={args.workers})")
    executor = SlurmPipelineExecutor(
        job_name="datatrove-bucket-test",
        pipeline=[
            docs,
            ParquetWriter(output_folder=output_url),
        ],
        tasks=args.tasks,
        workers=args.workers,
        time="00:20:00",
        partition=args.partition,
        qos=args.qos,
        logging_dir=logs_url,
        cpus_per_task=2,
        mem_per_cpu_gb=2,
    )
    executor.run()

    fs = HfFileSystem()
    shards = fs.find(f"buckets/{args.bucket}/v1/parquet", maxdepth=2)
    if len(shards) != args.tasks:
        print(f"FAIL: expected {args.tasks} parquet shards in bucket, got {len(shards)}")
        return 1

    reader = ParquetReader(data_folder=output_url)
    read_back = list(reader())
    if len(read_back) != len(docs):
        print(f"FAIL: read {len(read_back)} docs (expected {len(docs)})")
        return 1

    delete_bucket(args.bucket, missing_ok=True)
    print(f"PASS: Slurm pipeline wrote {len(docs)} docs across {args.tasks} shards to {args.bucket}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
