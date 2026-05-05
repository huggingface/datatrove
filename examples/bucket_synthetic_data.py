"""Four ways to use Hugging Face storage buckets with datatrove.

Buckets (``hf://buckets/<org>/<bucket>``) are S3-like, mutable object storage
backed by Xet. They are the recommended destination for raw and intermediate
data in synthetic-data pipelines; promote to a Git-based dataset
(``hf://datasets/<org>/<dataset>``) once the data is ready to be published.

Run this file directly to see each approach printed; uncomment the
``executor.run()`` call you want to actually execute (you need an HF token with
bucket-write access).
"""

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import HuggingFaceBucketWriter, ParquetWriter


ORG = "my-org"
BUCKET = "synth-data"
PREFIX = "v1"


# Approach A -- HuggingFaceBucketWriter (large datasets, staged Xet upload).
# Files are written locally first, then pushed to the bucket on rotation /
# close via ``batch_bucket_files``. Auto-creates the bucket on first upload.
def make_executor_with_bucket_writer() -> LocalPipelineExecutor:
    return LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=f"hf://buckets/{ORG}/{BUCKET}/raw/"),
            LambdaFilter(lambda doc: len(doc.text) > 100),
            HuggingFaceBucketWriter(
                bucket=f"{ORG}/{BUCKET}",
                prefix=f"{PREFIX}/filtered",
                private=True,
                cleanup=True,
                overwrite=True,  # replace existing files at prefix (default: False = append)
            ),
        ],
        tasks=8,
        workers=4,
        logging_dir=f"hf://buckets/{ORG}/{BUCKET}/logs/{PREFIX}",
    )


# Approach B -- Direct fsspec paths (simple, small/medium datasets).
# ParquetWriter just opens an ``hf://buckets/...`` URL via HfFileSystem. Good
# for ad-hoc use; use Approach A when you need staged uploads or auto-create.
def make_executor_with_fsspec_path() -> LocalPipelineExecutor:
    return LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=f"hf://buckets/{ORG}/{BUCKET}/raw/"),
            LambdaFilter(lambda doc: len(doc.text) > 100),
            ParquetWriter(output_folder=f"hf://buckets/{ORG}/{BUCKET}/{PREFIX}/filtered/"),
        ],
        tasks=8,
        workers=4,
        logging_dir=f"hf://buckets/{ORG}/{BUCKET}/logs/{PREFIX}",
    )


# Approach C -- hf-mount (zero code changes, best read performance).
# Mount the bucket once with the external Rust tool:
#   $ hf-mount start bucket my-org/synth-data /mnt/synth-data
# Then point datatrove at plain local paths.
def make_executor_with_hf_mount() -> LocalPipelineExecutor:
    return LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder="/mnt/synth-data/raw/"),
            LambdaFilter(lambda doc: len(doc.text) > 100),
            ParquetWriter(output_folder=f"/mnt/synth-data/{PREFIX}/filtered/"),
        ],
        tasks=8,
        workers=4,
        logging_dir=f"/mnt/synth-data/logs/{PREFIX}",
    )


# Approach D -- HF Jobs volume mounts (zero setup on HF infra).
# Submit the job with a ``-v`` mount; the bucket is exposed at ``/data``:
#   $ hf jobs run -v hf://buckets/my-org/synth-data:/data \
#       python:3.12 python pipeline.py
def make_executor_with_hf_jobs_mount() -> LocalPipelineExecutor:
    return LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder="/data/raw/"),
            LambdaFilter(lambda doc: len(doc.text) > 100),
            ParquetWriter(output_folder=f"/data/{PREFIX}/filtered/"),
        ],
        tasks=8,
        workers=4,
        logging_dir=f"/data/logs/{PREFIX}",
    )


if __name__ == "__main__":
    print("Approach A -- HuggingFaceBucketWriter (staged Xet upload):")
    print(make_executor_with_bucket_writer())
    print("\nApproach B -- ParquetWriter on hf://buckets/... (direct fsspec):")
    print(make_executor_with_fsspec_path())
    print("\nApproach C -- hf-mount + plain local paths:")
    print(make_executor_with_hf_mount())
    print("\nApproach D -- HF Jobs volume mount:")
    print(make_executor_with_hf_jobs_mount())
    # Uncomment to actually run one of them (requires HF token):
    # make_executor_with_bucket_writer().run()
