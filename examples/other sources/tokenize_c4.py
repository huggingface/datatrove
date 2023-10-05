from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.io import LocalInputDataFolder, S3InputDataFolder, S3OutputDataFolder
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


def format_adapter(d: dict, path: str, li: int):
    import os.path

    return {
        "content": d["text"],
        "data_id": f"{os.path.splitext(os.path.basename(path))[0]}_{li}",
        "metadata": {
            "timestamp": d["timestamp"],
            "url": d["url"],
        },
    }


DATASET_NAME = "c4"
LOCAL_INPUT_READER = JsonlReader(
    LocalInputDataFolder("/fsx/guilherme/c4/c4/en", extension=".json.gz"), compression="gzip", adapter=format_adapter
)

dist_executor = SlurmPipelineExecutor(
    job_name="c4_tok_1",
    pipeline=[
        LOCAL_INPUT_READER,
        DocumentTokenizer(
            S3OutputDataFolder(
                path=f"s3://extreme-scale-datasets/{DATASET_NAME}/tokenized/",
                local_path=f"/scratch/guilherme/{DATASET_NAME}/tokenized/",
            ),
            save_filename=f"{DATASET_NAME}_tokenized",
        ),
    ],
    tasks=1001,
    workers=64,
    time="72:00:00",
    partition="production-cluster",
    logging_dir=f"/fsx/guilherme/logs/tokenize_{DATASET_NAME}",
)
dist_executor.run()

merge_executor = SlurmPipelineExecutor(
    job_name="c4_tok_2",
    pipeline=[
        DocumentTokenizerMerger(
            input_folder=S3InputDataFolder(
                path=f"s3://extreme-scale-datasets/{DATASET_NAME}/tokenized/",
                local_path=f"/scratch/guilherme/{DATASET_NAME}/tokenized/",
            ),
            output_folder=S3OutputDataFolder(
                path=f"s3://extreme-scale-datasets/{DATASET_NAME}/standard/",
                local_path=f"/scratch/guilherme/{DATASET_NAME}/standard/",
            ),
            save_filename=f"{DATASET_NAME}",
        ),
    ],
    tasks=1,
    time="50:00:00",
    partition="production-cluster",
    logging_dir=f"/fsx/guilherme/logs/tokenize_{DATASET_NAME}_merged",
    mem_per_cpu_gb=11,
    depends=dist_executor,
)
merge_executor.run()
