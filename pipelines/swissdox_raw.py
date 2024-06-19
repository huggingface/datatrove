"""

"""
from swiss_ai.readers.swissdox import RawSwissDoxReader
from datatrove.pipeline.tokens import TokensCounter, LengthCounter
from swiss_ai.writers.jsonl import SwissAIJsonlWriter
from datatrove.executor.local import LocalPipelineExecutor
import os

os.environ["HF_BASE"] = "/work_space_data/hf_cache/"
if not os.path.exists("/work_space_data/swissdox"):
    os.makedirs("/work_space_data/swissdox")

if __name__ == "__main__":
    pipeline = [
        RawSwissDoxReader(
            data_folder="/work_space_data/swissdox",
            limit=-1
        ),
        LengthCounter(),
        TokensCounter(tokenizer_name_or_path='t5-small'),
        SwissAIJsonlWriter(
            output_folder="/work_space_data/swissdox/jsonl"
        )
    ]

    exec = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=64,
        workers=16,
        start_method="spawn",
        logging_dir="/work_space_data/swissdox/logging",
    )

    exec.run()
