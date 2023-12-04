from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.io import S3InputDataFolder, S3OutputDataFolder
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens.counter import LengthCounter


# data to dedup: s3://extreme-scale-data-processing/early_pipeline/output/
# place to save temporary data that can be deleted after dedup: s3://extreme-scale-dp-temp/early_web_pipeline/minhash
# place where deduped (output) data should be saved: s3://extreme-scale-data-processing/early_pipeline/minhash/

# configurable options here

BASE_LOGS_PATH = "/fsx/thomwolf/logs_2/early_pipeline_dp/doc_length_counter/"
S3_OUTPUT_BASE_PATH = "s3://extreme-scale-data-processing/early_pipeline/doc_length_counter"

count_proc = SlurmPipelineExecutor(
    job_name="get_doc_length",
    pipeline=[
        JsonlReader(
            S3InputDataFolder(
                "s3://fineweb-data-processing/base_processing/dedup/minhash/output/CC-MAIN-2023-14/", stream=True
            ),
            # S3InputDataFolder(f"s3://extreme-scale-data-processing/early_pipeline/minhash_nov/output", stream=True),
        ),
        LengthCounter(),
    ],
    tasks=500,
    # workers=1001,
    time="72:00:00",
    partition="production-cluster",
    skip_completed=False,
    logging_dir=S3OutputDataFolder(path=f"{S3_OUTPUT_BASE_PATH}/logs", local_path=f"{BASE_LOGS_PATH}/logs"),
    slurm_logs_folder=f"{BASE_LOGS_PATH}/slurm_logs",
)

count_proc.run()
