from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import LocalOutputDataFolder, S3InputDataFolder, S3OutputDataFolder
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


pipeline = [
    JsonlReader(
        S3InputDataFolder("s3://bigcode-experiments/the-pile-sharded/", stream=True),
        gzip=False,
        content_key="text",
        limit=100,
    ),
    DocumentTokenizer(
        LocalOutputDataFolder(path="/home/gui/hf_dev/datatrove/examples_test/piletokenized-test/tokenized"),
        save_filename="the_pile_tokenized",
    ),
]

executor: PipelineExecutor = LocalPipelineExecutor(
    pipeline=pipeline,
    tasks=20,
    workers=20,
    skip_completed=False,
    logging_dir=S3OutputDataFolder(
        "s3://extreme-scale-dp-temp/logs/tests/piletokenized/tokenized",
        local_path="/home/gui/hf_dev/datatrove/examples_test/piletokenized-test/logs",
        cleanup=False,
    ),
)
executor.run()
