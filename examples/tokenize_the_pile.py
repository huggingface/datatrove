import os.path

from datatrove.executor.base import PipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.io import LocalOutputDataFolder, S3InputDataFolder
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


def format_adapter(d: dict, path: str, li: int):
    return {
        "content": d["text"],
        "data_id": f"{os.path.splitext(os.path.basename(path))[0]}_{li}",
        "metadata": d["meta"],
    }


pipeline = [
    JsonlReader(
        S3InputDataFolder("s3://bigcode-experiments/the-pile-sharded/", stream=True),
        gzip=False,
        adapter=format_adapter,
    ),
    DocumentTokenizer(
        LocalOutputDataFolder(path="/fsx/guilherme/the-pile/tokenized"), save_filename="the_pile_tokenized"
    ),
]

executor: PipelineExecutor = SlurmPipelineExecutor(
    pipeline=pipeline,
    tasks=20,
    time="12:00:00",
    partition="dev-cluster",
    logging_dir="/fsx/guilherme/logs/tokenize_the_pile",
)
executor.run()
