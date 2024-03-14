from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


DATASET_NAME = "c4"


dist_executor = SlurmPipelineExecutor(
    job_name="c4_tok_1",
    pipeline=[
        JsonlReader(
            "hf://datasets/allenai/c4/en/",  # read directly from huggingface
            glob_pattern="c4-*.json.gz",  # only get the english files
        ),
        DocumentTokenizer(
            output_folder=f"s3://extreme-scale-datasets/{DATASET_NAME}/tokenized/",
            local_working_dir=f"/scratch/guilherme/{DATASET_NAME}/tokenized/",
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
            input_folder=f"s3://extreme-scale-datasets/{DATASET_NAME}/tokenized/",
            output_folder=f"s3://extreme-scale-datasets/{DATASET_NAME}/standard/",
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
