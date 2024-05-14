"""

"""

import sys, os, re

sys.path.append("../src")
from datatrove.pipeline.readers.huggingface import HuggingFaceDatasetReader
from datatrove.pipeline.tokens import TokensCounter, LengthCounter
from swiss_ai.writers.jsonl import SwissAIJsonlWriter
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)

# os.environ["HF_BASE"] = "/work_space_data/hf_cache"
# Don't forget to set the HF_BASE environment variable to a valid path


def find_years(text):
    # Regex pattern to match four-digit numbers that are likely to be years
    # This pattern matches any number from 1900 to 2099
    pattern = r"\b(19[0-9]{2}|20[0-9]{2})\b"

    # Find all matches in the text
    years = re.findall(pattern, text)

    return years


def _multilegal_adapter(data: dict, path: str, id_in_file: int | str):
    years = find_years(data["text"])
    if len(years) > 0:
        # very crude estimation of the year..
        year = max(int(year) for year in years if int(year) <= 2024)
    else:
        year = 2024
    metadata = {
        "language": data["language"],
        "year": year,
        "optional": {"type": data["type"], "jurisdiction": data["jurisdiction"]},
    }

    return {
        "text": data.pop("text", ""),
        "id": f"{path}/{id_in_file}",
        "media": data.pop("media", []),
        "metadata": metadata,
    }


if __name__ == "__main__":
    INPUT_READER = HuggingFaceDatasetReader(
        dataset="joelniklaus/Multi_Legal_Pile",
        dataset_options={
            "split": "train",
            "name": "all_legislation",
            "cache_dir": os.environ["HF_BASE"],
            "trust_remote_code": True,
        },
        progress=True,
        adapter=_multilegal_adapter,
        limit=1000,
    )

    main_processing_executor = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            TokensCounter(),
            LengthCounter(),
            SwissAIJsonlWriter(
                output_folder=f'/{os.environ["HF_BASE"]}/multilegal_pile/jsonl'
            ),
        ],
        tasks=16,
        workers=1,
        start_method="spawn",
        logging_dir=f'/{os.environ["HF_BASE"]}/multilegal_pile/logging',
    )

    main_processing_executor.run()

    minhash_config = MinhashConfig(
        use_64bit_hashes=True
    )  # better precision -> fewer false positives (collisions)
    TOTAL_TASKS = 1000
    stage1 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f'{os.environ["HF_BASE"]}/multilegal_pile/signatures',
                config=minhash_config,
            ),
        ],
        tasks=TOTAL_TASKS,
        time="5:00:00",
        partition="hopper-cpu",
        logging_dir=f'/{os.environ["HF_BASE"]}/multilegal_pile/logging/signatures',
        slurm_logs_folder=f'/{os.environ["HF_BASE"]}/multilegal_pile/logging/signatures/slurm_logs',
        randomize_start=True,
        depends=main_processing_executor,  # only start after the first one completes
    )
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f'{os.environ["HF_BASE"]}/multilegal_pile/signatures',
                output_folder=f'{os.environ["HF_BASE"]}/multilegal_pile/buckets',
                config=MinhashConfig(use_64bit_hashes=True),
            ),
        ],
        tasks=minhash_config.num_buckets
        * 50,  # the code supports parallelizing each bucket. here we run 50
        # workers per bucket
        randomize_start=True,
        logging_dir=f'{os.environ["HF_BASE"]}/multilegal_pile/logging/buckets',
        partition="hopper-cpu",
        time="02:00:00",
        mem_per_cpu_gb=4,
        cpus_per_task=3,  # you can add run more (smaller) tasks if you do not have a lot of memory
        depends=stage1,
    )
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f'{os.environ["HF_BASE"]}/multilegal_pile/buckets',
                output_folder=f'{os.environ["HF_BASE"]}/multilegal_pile/remove_ids',
                config=minhash_config,
            ),
        ],
        tasks=1,  # this step runs on a single task
        logging_dir=f'{os.environ["HF_BASE"]}/multilegal_pile/clustering',
        partition="hopper-cpu",
        time="30:00:00",  # and can also be quite slow. Usually not this slow though
        mem_per_cpu_gb=25,
        cpus_per_task=8,  # if you dedup a full dump, you do need a lot of memory for this one
        depends=stage2,
    )

    stage4 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            TokensCounter(),  # you can remove this one, it's just a nice way to know how many tokens we have
            # before and after dedup
            MinhashDedupFilter(
                input_folder=f'{os.environ["HF_BASE"]}/multilegal_pile//remove_ids'
            ),
            # run the PII removal
            # PIIFormatter(),
            SwissAIJsonlWriter(
                output_folder=f'/{os.environ["HF_BASE"]}/multilegal_pile/jsonl'
            ),
        ],
        tasks=TOTAL_TASKS,
        logging_dir=f'{os.environ["HF_BASE"]}/multilegal_pile/filtering',
        partition="hopper-cpu",
        time="5:00:00",
        mem_per_cpu_gb=4,
        depends=stage3,
    )

    # launch dedup pipelines
    stage4.run()
