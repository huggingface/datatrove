from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.io import S3InputDataFolder, S3OutputDataFolder
from datatrove.pipeline.filters import ListFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


# data to dedup: s3://extreme-scale-data-processing/early_pipeline/output/
# place to save temporary data that can be deleted after dedup: s3://extreme-scale-dp-temp/early_web_pipeline/minhash
# place where deduped (output) data should be saved: s3://extreme-scale-data-processing/early_pipeline/minhash/

# configurable options here

KEEP_SMALL = False

EXP = "small" if KEEP_SMALL else "large"

BASE_LOGS_PATH = f"/fsx/thomwolf/logs_2/early_pipeline_dp/test_{EXP}_text"
S3_TEMP_BASE_PATH = f"s3://extreme-scale-dp-temp/early_web_pipeline/test_{EXP}_text"
LOCAL_TEMP_BASE_PATH = f"/scratch/thomwolf/early_web_pipeline/test_{EXP}_text"
S3_OUTPUT_BASE_PATH = f"s3://extreme-scale-data-processing/early_pipeline/test_{EXP}_text"


def small_length_filter(data, rank, world_size):
    for sample in data:
        if sample.metadata["token_count"] < 450:
            yield sample  # keep this one


def large_length_filter(data, rank, world_size):
    for sample in data:
        if sample.metadata["token_count"] >= 450:
            yield sample  # keep this one


dist_tok = SlurmPipelineExecutor(
    job_name=f"tok_1_{EXP}_text",
    pipeline=[
        JsonlReader(
            S3InputDataFolder(
                "s3://fineweb-data-processing/base_processing/dedup/minhash/output/CC-MAIN-2023-14/", stream=True
            ),
            # S3InputDataFolder(f"s3://extreme-scale-data-processing/early_pipeline/minhash_nov/output", stream=True),
        ),
        ListFilter(),
        small_length_filter if KEEP_SMALL else large_length_filter,
        DocumentTokenizer(
            S3OutputDataFolder(
                path=f"{S3_TEMP_BASE_PATH}/tokenized",
                local_path=f"{LOCAL_TEMP_BASE_PATH}/tokenized",
            ),
            save_filename=f"test_{EXP}_text_tokenized",
        ),
    ],
    tasks=500,
    # workers=1001,
    time="72:00:00",
    partition="production-cluster",
    logging_dir=f"{BASE_LOGS_PATH}/tokenized",
    slurm_logs_folder=f"{BASE_LOGS_PATH}/tokenized/slurm_logs",
)

merge_tok = SlurmPipelineExecutor(
    job_name=f"tok_2_{EXP}_text",
    pipeline=[
        DocumentTokenizerMerger(
            input_folder=S3InputDataFolder(
                path=f"{S3_TEMP_BASE_PATH}/tokenized",
                local_path=f"{LOCAL_TEMP_BASE_PATH}/tokenized",
            ),
            output_folder=S3OutputDataFolder(
                path=f"s3://extreme-scale-datasets/early_web_pipeline/test_{EXP}_text/",
                local_path=f"/scratch/thomwolf/early_web_pipeline/test_{EXP}_text/merged",
            ),
            save_filename=f"test_{EXP}_text_tokenized",
        ),
    ],
    tasks=1,
    time="50:00:00",
    partition="production-cluster",
    logging_dir=f"{BASE_LOGS_PATH}/merged_toks",
    slurm_logs_folder=f"{BASE_LOGS_PATH}/merged_toks/slurm_logs",
    mem_per_cpu_gb=11,
    cpus_per_task=30,
    depends=dist_tok,
)

merge_tok.run()
