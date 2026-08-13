"""
MinHash deduplication on Hugging Face Jobs — the `JobsPipelineExecutor` twin of
`minhash_deduplication.py` (Slurm). The 4-stage pipeline is identical; only the executors change.

Note: `JobsPipelineExecutor` is experimental and may change, break, or be removed in future releases.

  stage1 signatures -> stage2 buckets (depends=1) -> stage3 cluster (depends=2) -> stage4 filter (depends=3)

Notes specific to running this multi-stage dedup on Jobs (see `filter_hf_dataset_jobs.py` for the
general Slurm->Jobs comparison):
- `MINHASH_BASE_PATH` and the logs must be a shared REMOTE path that all four stages read/write.
  HF Storage Buckets (`hf://buckets/<user>/<bucket>/...`, needs huggingface_hub>=1.6.0) are the best
  fit: the same fsspec code path as `hf://datasets/...`, but object-storage semantics — no per-repo
  commit queue and no read-after-write lag between stages.
- At higher fan-out, many Jobs committing small files to a single hf:// *dataset repo* can hit per-repo
  commit rate limits (429 "maximum time in concurrency queue reached") in the signature stage — a
  signature stage that failed 6/24 ranks on a dataset repo at `workers=8` passed 24/24 on a bucket with
  identical settings. If you must stay on a dataset repo: lower `workers`, and/or rerun to resume —
  only the failed ranks are relaunched.
- stage 2 sets `lines_to_buffer=-1` on `MinhashDedupBuckets`, and on remote storage you should too
  (at this kind of scale). The default (5) is a RAM bound tuned for local filesystems: stage 2 holds
  EVERY stage-1 signature file open at once for its merge, and the value also sets the fsspec
  `block_size` (5 lines ≈ 340 bytes) — so over hf:// each couple of reads becomes its own HTTP range
  request, ~500 requests per signature file (measured: this stage went 65 min → 99 s after switching
  to -1). -1 instead reads each file in a single request but buffers whole files in RAM: budget
  `num_stage1_tasks × sig_file_size` (here 24 × ~340 KB ≈ 8 MB — trivial). At thousands of stage-1
  tasks, keep a bounded `lines_to_buffer` and stage the bucket's signature files to the Job's local
  ephemeral disk first instead.
- `dependencies` = `datatrove[io,processing]` PLUS `spacy`: the signature stage's English word tokenizer
  is `spacy.blank("en")` (no model download), and `spacy` only ships in the heavy `multilingual` extra,
  so add the bare package rather than that whole extra.
- `max_retries >= 1` matters on `hf://datasets/` paths: dataset repos have read-after-write lag, so a
  stage can 404 on a file just written — even its own (stage 1 re-reading a `.sig` it wrote milliseconds
  earlier has been observed to 404); a retry succeeds once it propagates. Buckets and other strongly
  consistent stores don't need it, but it's cheap insurance either way.
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

# a shared REMOTE base path all four stages use — never a local path. A Storage Bucket is the
# recommended default (see module docstring); an hf://datasets/ repo works but is slower and
# flakier for this inter-stage traffic.
MINHASH_BASE_PATH = "hf://buckets/my_org/my-minhash/data"
LOGS_FOLDER = "hf://buckets/my_org/my-minhash/logs"

# datatrove + processing (nltk/xxhash/regex/tokenizers) + bare spacy (English word tokenizer).
# The Job needs datatrove>=0.10.0, the release that introduced JobsPipelineExecutor:
# with an older one it dies at unpickle time (the class doesn't exist in its env).
DEPENDENCIES = ["datatrove[io,processing]", "spacy"]

TOTAL_TASKS = 50

# this is the original data that we want to deduplicate — stage 1 and stage 4 must use the SAME reader
INPUT_READER = HuggingFaceDatasetReader("stanfordnlp/imdb", dataset_options={"split": "train"}, text_key="text")

# shared executor config for every stage
common = {
    "flavor": "cpu-basic",
    "dependencies": DEPENDENCIES,
    "timeout": "2h",
    "max_retries": 2,  # cheap insurance; required on hf://datasets/ paths (read-after-write 404s, see docstring)
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
            # -1 = read each signature file in ONE request instead of one per ~340 bytes (65 min → 99 s
            # for this stage). RAM tradeoff + when NOT to use -1: see the note in the module docstring.
            lines_to_buffer=-1,
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
