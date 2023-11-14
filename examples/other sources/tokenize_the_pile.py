from datatrove.executor.base import PipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.io import LocalInputDataFolder, LocalOutputDataFolder, S3InputDataFolder, S3OutputDataFolder
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


pipeline = [
    JsonlReader(
        S3InputDataFolder("s3://bigcode-experiments/the-pile-sharded/", stream=True),
        content_key="text",
        limit=100,
    ),
    DocumentTokenizer(
        LocalOutputDataFolder(path="/fsx/guilherme/tests/piletokenized-test/tokenized"),
        save_filename="the_pile_tokenized",
    ),
]

executor: PipelineExecutor = SlurmPipelineExecutor(
    pipeline=pipeline,
    tasks=2,
    workers=1,
    skip_completed=False,
    logging_dir=S3OutputDataFolder(
        "s3://extreme-scale-dp-temp/logs/tests/piletokenized/tokenized",
        local_path="/fsx/guilherme/tests/piletokenized-test/logs",
        cleanup=False,
    ),
    slurm_logs_folder="/fsx/guilherme/tests/piletokenized-test/slurm_logs",
    partition="production-cluster",
    time="01:00:00",
)
executor.run()

executor2: PipelineExecutor = SlurmPipelineExecutor(
    pipeline=[
        DocumentTokenizerMerger(
            input_folder=LocalInputDataFolder(path="/fsx/guilherme/tests/piletokenized-test/tokenized"),
            output_folder=LocalOutputDataFolder(path="/fsx/guilherme/tests/piletokenized-test/merged"),
            save_filename="hello",
            max_tokens_per_file=100000,
        )
    ],
    depends=executor,
    tasks=1,
    skip_completed=False,
    logging_dir=S3OutputDataFolder(
        "s3://extreme-scale-dp-temp/logs/tests/piletokenized/merged",
        local_path="/fsx/guilherme/tests/piletokenized-test/logs2",
        cleanup=False,
    ),
    slurm_logs_folder="/fsx/guilherme/tests/piletokenized-test/slurm_logs2",
    partition="production-cluster",
    time="01:00:00",
)
executor2.run()
