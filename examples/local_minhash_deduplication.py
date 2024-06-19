from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter

# you can also change ngrams or the number of buckets and their size here
minhash_config = MinhashConfig(use_64bit_hashes=True)  # better precision -> fewer false positives (collisions)

corpus = 'curiavista'

S3_MINHASH_BASE_PATH = f"/work_space_data/{corpus}/minhash/"

S3_LOGS_FOLDER = f"/work_space_data/{corpus}/minhash/logging"
LOCAL_LOGS_FOLDER = f"/work_space_data/{corpus}/dedup/logging"

TOTAL_TASKS = 16
WORKERS = 16
# this is the original data that we want to deduplicate
INPUT_READER = JsonlReader(f"/work_space_data/{corpus}/jsonl", compression='gzip', progress=True)

if __name__ == '__main__':
    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            #LambdaFilter(lambda doc: not doc.metadata.get('delete', False)),
            MinhashDedupSignature(output_folder=f"{S3_MINHASH_BASE_PATH}/signatures", config=minhash_config),
        ],
        tasks=TOTAL_TASKS,
        workers=WORKERS,
        start_method="spawn",
        logging_dir=f"{S3_LOGS_FOLDER}/signatures",
    )

    # stage 2 finds matches between signatures in each bucket
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{S3_MINHASH_BASE_PATH}/signatures",
                output_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        logging_dir=f"{S3_LOGS_FOLDER}/buckets",
        depends=stage1,
        start_method="spawn",
        workers=WORKERS,
    )

    # stage 3 creates clusters of duplicates using the results from all buckets
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
                output_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,
        logging_dir=f"{S3_LOGS_FOLDER}/clusters",
        depends=stage2,
        start_method="spawn",
        workers=WORKERS
    )

    # stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
    # the data must match exactly stage 1, so number of tasks and the input source must be the same
    stage4 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            #LambdaFilter(lambda doc: not doc.metadata.get('delete', False)),
            TokensCounter(),  # nice way to see how many tokens we had before and after deduplication
            MinhashDedupFilter(
                input_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids",
                exclusion_writer=JsonlWriter(f"{S3_MINHASH_BASE_PATH}/removed"),
            ),
            JsonlWriter(output_folder=f"{S3_MINHASH_BASE_PATH}/deduplicated_output"),
        ],
        tasks=TOTAL_TASKS,
        logging_dir=f"{S3_LOGS_FOLDER}/filter",
        depends=stage3,
        start_method="spawn",
        workers=WORKERS
    )

    stage4.run()
