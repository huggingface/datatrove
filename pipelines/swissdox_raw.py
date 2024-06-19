"""

"""
from swiss_ai.readers.swissdox import RawSwissDoxReader
from datatrove.pipeline.tokens import TokensCounter, LengthCounter
from swiss_ai.writers.jsonl import SwissAIJsonlWriter
from datatrove.executor.local import LocalPipelineExecutor


if __name__ == '__main__':
    pipeline = [
        RawSwissDoxReader(
            data_folder="/work_space_data/swissdox",
            limit=-1
        ),
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
        logging_dir="/work_space_data/swissdox/logging"
    )

    exec.run()