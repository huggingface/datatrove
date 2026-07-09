"""
MinHash deduplication on Hugging Face Jobs — the `JobsPipelineExecutor` twin of
`minhash_deduplication.py` (Slurm). The 4-stage pipeline is identical; only the executors change.

Note: `JobsPipelineExecutor` is experimental and its API may change in future releases.

  stage1 signatures -> stage2 buckets (depends=1) -> stage3 cluster (depends=2) -> stage4 filter (depends=3)

Notes specific to running this multi-stage dedup on Jobs (see `filter_hf_dataset_jobs.py` for the
general Slurm->Jobs comparison):
- `MINHASH_BASE_PATH` and the logs must be a shared REMOTE path (hf:// or s3://) that all four stages
  read/write. HF S3-compatible buckets are a good fit and reduce read-after-write races between stages.
- `dependencies` = `datatrove[io,processing]` PLUS `spacy`: the signature stage's English word tokenizer
  is `spacy.blank("en")` (no model download), and `spacy` only ships in the heavy `multilingual` extra,
  so add the bare package rather than that whole extra.
- `max_retries >= 1` matters here: `hf://` has read-after-write lag, so a downstream stage can 404 on a
  file the previous stage just wrote; a retry succeeds once it propagates. (Not needed on a strongly
  consistent store.)
- stage 1 and stage 4 MUST use the same input reader and the same task count (unchanged from Slurm).
- Set `workers=N` on the stages to cap concurrent Jobs (the analog of Slurm's `%workers`).
"""

from datatrove.executor import JobsPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages


# you can also change ngrams or the number of buckets and their size here
minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
)  # better precision -> fewer false positives (collisions)

# a shared REMOTE base path all four stages use (hf:// or s3://) — never a local path
MINHASH_BASE_PATH = "hf://datasets/my_org/my-minhash"
LOGS_FOLDER = "hf://datasets/my_org/my-minhash-logs"

# datatrove + processing (nltk/xxhash/regex/tokenizers) + bare spacy (English word tokenizer).
# Until JobsPipelineExecutor is released, install datatrove from your branch, e.g.:
#   "datatrove[io,processing] @ git+https://github.com/<user>/datatrove@<branch>"
DEPENDENCIES = ["datatrove[io,processing]", "spacy"]

TOTAL_TASKS = 50

# this is the original data that we want to deduplicate — stage 1 and stage 4 must use the SAME reader
INPUT_READER = HuggingFaceDatasetReader("stanfordnlp/imdb", dataset_options={"split": "train"}, text_key="text")

# shared executor config for every stage
common = {
    "flavor": "cpu-basic",
    "dependencies": DEPENDENCIES,
    "timeout": "2h",
    "max_retries": 2,  # tolerate transient hf:// read-after-write 404s between stages
}

# stage 1 computes minhash signatures for each task (each task gets a set of files)
stage1 = JobsPipelineExecutor(
    job_name="mh1",
    pipeline=[
        INPUT_READER,
        MinhashDedupSignature(
            output_folder=f"{MINHASH_BASE_PATH}/signatures", config=minhash_config, language=Languages.english
        ),
    ],
    tasks=TOTAL_TASKS,
    logging_dir=f"{LOGS_FOLDER}/signatures",
    **common,
)

# stage 2 finds matches between signatures in each bucket
stage2 = JobsPipelineExecutor(
    job_name="mh2",
    pipeline=[
        MinhashDedupBuckets(
            input_folder=f"{MINHASH_BASE_PATH}/signatures",
            output_folder=f"{MINHASH_BASE_PATH}/buckets",
            config=minhash_config,
        ),
    ],
    tasks=minhash_config.num_buckets,
    logging_dir=f"{LOGS_FOLDER}/buckets",
    depends=stage1,
    **common,
)

# stage 3 creates clusters of duplicates using the results from all buckets
stage3 = JobsPipelineExecutor(
    job_name="mh3",
    pipeline=[
        MinhashDedupCluster(
            input_folder=f"{MINHASH_BASE_PATH}/buckets",
            output_folder=f"{MINHASH_BASE_PATH}/remove_ids",
            config=minhash_config,
        ),
    ],
    tasks=1,
    logging_dir=f"{LOGS_FOLDER}/clusters",
    depends=stage2,
    **common,
)

# stage 4 reads the original input and removes all but 1 sample per duplicate cluster
# the data must match exactly stage 1, so number of tasks and the input source must be the same
stage4 = JobsPipelineExecutor(
    job_name="mh4",
    pipeline=[
        INPUT_READER,
        TokensCounter(),  # nice way to see how many tokens we had before and after deduplication
        MinhashDedupFilter(
            input_folder=f"{MINHASH_BASE_PATH}/remove_ids",
            exclusion_writer=JsonlWriter(f"{MINHASH_BASE_PATH}/removed"),
        ),
        JsonlWriter(output_folder=f"{MINHASH_BASE_PATH}/deduplicated_output"),
    ],
    tasks=TOTAL_TASKS,
    logging_dir=f"{LOGS_FOLDER}/filter",
    depends=stage3,
    **common,
)


if __name__ == "__main__":
    stage4.run()
