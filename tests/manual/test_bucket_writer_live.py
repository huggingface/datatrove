"""Live end-to-end test for ``HuggingFaceBucketWriter``.

Run manually:

    python tests/manual/test_bucket_writer_live.py --bucket myorg/test-datatrove-bucket

Requires ``HF_TOKEN`` with bucket-write scope. The bucket is created if missing
and deleted on success.
"""

from __future__ import annotations

import argparse
import getpass
import sys
import uuid

from huggingface_hub import delete_bucket

from datatrove.data import Document
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers import HuggingFaceBucketWriter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default=f"{getpass.getuser()}/test-datatrove-{uuid.uuid4().hex[:8]}",
        help="Bucket id to use for the test (will be auto-created).",
    )
    parser.add_argument("--n_docs", type=int, default=100)
    parser.add_argument("--prefix", default="round-trip")
    args = parser.parse_args()

    docs = [Document(text=f"hello world {i}", id=f"doc-{i}", metadata={"score": float(i)}) for i in range(args.n_docs)]

    print(f"Writing {args.n_docs} docs to bucket={args.bucket} prefix={args.prefix}")
    writer = HuggingFaceBucketWriter(
        bucket=args.bucket,
        prefix=args.prefix,
        private=True,
        cleanup=True,
    )
    with writer:
        for doc in docs:
            writer.write(doc)

    bucket_url = f"hf://buckets/{args.bucket}/{args.prefix}/"
    print(f"Reading back from {bucket_url}")
    reader = ParquetReader(data_folder=bucket_url)
    read_back = list(reader())

    ok = len(read_back) == args.n_docs and all(
        rd.text == orig.text and rd.id == orig.id for rd, orig in zip(read_back, docs)
    )

    if ok:
        delete_bucket(args.bucket, missing_ok=True)
        print(f"PASS: round-trip of {args.n_docs} docs through bucket {args.bucket}")
        return 0

    print(f"FAIL: read back {len(read_back)} docs (expected {args.n_docs})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
