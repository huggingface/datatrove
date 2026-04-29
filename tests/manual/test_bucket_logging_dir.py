"""Live test for using ``hf://buckets/...`` as ``logging_dir``.

Run manually:

    python tests/manual/test_bucket_logging_dir.py --bucket myorg/test-datatrove-logs

Verifies that ``LocalPipelineExecutor`` can stream completion files, stats and
``stats.json`` to a bucket-backed ``logging_dir``, and that ``skip_completed``
correctly short-circuits a re-run.
"""

from __future__ import annotations

import argparse
import getpass
import sys
import tempfile
import uuid

from huggingface_hub import HfFileSystem, create_bucket, delete_bucket

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers.jsonl import JsonlWriter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default=f"{getpass.getuser()}/test-datatrove-{uuid.uuid4().hex[:8]}",
    )
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    # Bucket must exist before using it as logging_dir.
    create_bucket(args.bucket, private=True, exist_ok=True)

    logging_url = f"hf://buckets/{args.bucket}/logs/run1"
    print(f"Running pipeline with logging_dir={logging_url}")

    docs = [Document(text=f"hello {i}", id=f"doc-{i}") for i in range(50)]

    with tempfile.TemporaryDirectory() as out_dir:
        executor = LocalPipelineExecutor(
            pipeline=[
                docs,
                LambdaFilter(lambda doc: True),
                JsonlWriter(output_folder=out_dir),
            ],
            tasks=args.tasks,
            workers=args.workers,
            logging_dir=logging_url,
        )
        executor.run()

        # Confirm completions and stats made it to the bucket.
        fs = HfFileSystem()
        completions = fs.find(f"buckets/{args.bucket}/logs/run1/completions", maxdepth=2)
        stats = fs.find(f"buckets/{args.bucket}/logs/run1/stats", maxdepth=2)
        # stats/ contains per-task files plus a stats.json aggregate
        if len(completions) != args.tasks or len(stats) < args.tasks:
            print(
                f"FAIL: expected {args.tasks} completion files and >= {args.tasks} stats files; "
                f"got {len(completions)} completions, {len(stats)} stats"
            )
            return 1

        # Re-running with skip_completed=True should be a no-op.
        executor2 = LocalPipelineExecutor(
            pipeline=[docs, JsonlWriter(output_folder=out_dir)],
            tasks=args.tasks,
            workers=args.workers,
            logging_dir=logging_url,
            skip_completed=True,
        )
        executor2.run()

    delete_bucket(args.bucket, missing_ok=True)
    print(f"PASS: bucket-backed logging_dir works for {args.tasks} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
