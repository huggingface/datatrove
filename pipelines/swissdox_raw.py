"""

"""

from swiss_ai.readers.swissdox import RawSwissDoxReader
from datatrove.pipeline.tokens import TokensCounter, LengthCounter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor.local import LocalPipelineExecutor

os.environ["HF_BASE"] = "/work_space_data/hf_cache/"

if __name__ == '__main__':
    pipeline = [
        RawSwissDoxReader(
            data_folder="/work_space_data/swissdox",
            limit=-1
        ),
        TokensCounter(tokenizer_name_or_path='t5-small'),
        LengthCounter(),
        JsonlWriter(
            output_folder="/work_space_data/swissdox/jsonl"
        )
    ]

    exec = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=16,
        workers=16,
        start_method="spawn",
        logging_dir="/work_space_data/swissdox/logging"
    )

    exec.run()