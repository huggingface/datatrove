import sys
import os
import re
import psutil
import time
import threading
from datetime import datetime

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
from datatrove.pipeline.readers import JsonlReader

# os.environ["HF_BASE"] = "/work_space_data/hf_cache"
# Don't forget to set the HF_BASE environment variable to a valid path


def find_years(text):
    pattern = r"\b(19[0-9]{2}|20[0-9]{2})\b"
    years = re.findall(pattern, text)
    return years


def _multilegal_adapter(data: dict, path: str, id_in_file: int | str):
    years = find_years(data["text"])
    if len(years) > 0:
        try:
            year = max(int(year) for year in years if int(year) <= 2024)
        except:
            year = 2024
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


def log_resource_usage(log_file, stop_event):
    process = psutil.Process(os.getpid())
    with open(log_file, "a") as f:
        f.write(f"Resource usage log started at {datetime.now()}\n")
        while not stop_event.is_set():
            try:
                cpu_usage = process.cpu_percent(interval=1)
                memory_info = process.memory_info()
                memory_usage = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
                f.write(
                    f"{datetime.now()}: CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage} MB\n"
                )
                time.sleep(1)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                break
        f.write(f"Resource usage log ended at {datetime.now()}\n")


if __name__ == "__main__":
    log_file = "resource_usage_log.txt"
    stop_event = threading.Event()
    resource_log_thread = threading.Thread(
        target=log_resource_usage, args=(log_file, stop_event)
    )
    resource_log_thread.start()

    try:
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

        # main_processing_executor = LocalPipelineExecutor(
        #     pipeline=[
        #         INPUT_READER,
        #         TokensCounter(),
        #         LengthCounter(),
        #         SwissAIJsonlWriter(
        #             output_folder=f'/{os.environ["HF_BASE"]}/multilegal_pile/jsonl'
        #         ),
        #     ],
        #     tasks=16,
        #     workers=1,
        #     start_method="spawn",
        #     logging_dir=f'/{os.environ["HF_BASE"]}/multilegal_pile/logging',
        # )
        #
        # main_processing_executor.run()

        minhash_config = MinhashConfig(use_64bit_hashes=True)
        # TOTAL_TASKS = 16
        # stage1 = LocalPipelineExecutor(
        #     pipeline=[
        #         JsonlReader(
        #             data_folder=f'/{os.environ["HF_BASE"]}/multilegal_pile/jsonl'
        #         ),
        #         MinhashDedupSignature(
        #             output_folder=f'{os.environ["HF_BASE"]}/multilegal_pile/signatures',
        #             config=minhash_config,
        #         ),
        #     ],
        #     tasks=TOTAL_TASKS,
        #     logging_dir=f'/{os.environ["HF_BASE"]}/multilegal_pile/logging/signatures',
        #     # depends=main_processing_executor,
        # )
        # stage1.run()
        # stage2 = LocalPipelineExecutor(
        #     pipeline=[
        #         MinhashDedupBuckets(
        #             input_folder=f'{os.environ["HF_BASE"]}/multilegal_pile/signatures',
        #             output_folder=f'{os.environ["HF_BASE"]}/multilegal_pile/buckets',
        #             config=MinhashConfig(use_64bit_hashes=True),
        #         ),
        #     ],
        #     tasks=minhash_config.num_buckets * 50,
        #     logging_dir=f'{os.environ["HF_BASE"]}/multilegal_pile/logging/buckets',
        #     depends=stage1,
        # )
        # stage2.run()
        stage3 = LocalPipelineExecutor(
            pipeline=[
                MinhashDedupCluster(
                    input_folder=f'{os.environ["HF_BASE"]}/multilegal_pile/buckets',
                    output_folder=f'{os.environ["HF_BASE"]}/multilegal_pile/remove_ids',
                    config=minhash_config,
                ),
            ],
            tasks=1,
            logging_dir=f'{os.environ["HF_BASE"]}/multilegal_pile/clustering',
            # depends=stage2,
        )
        stage3.run()
        # stage4 = LocalPipelineExecutor(
        #     pipeline=[
        #         INPUT_READER,
        #         TokensCounter(),
        #         MinhashDedupFilter(
        #             input_folder=f'{os.environ["HF_BASE"]}/multilegal_pile/remove_ids'
        #         ),
        #         SwissAIJsonlWriter(
        #             output_folder=f'/{os.environ["HF_BASE"]}/multilegal_pile/jsonl'
        #         ),
        #     ],
        #     tasks=TOTAL_TASKS,
        #     logging_dir=f'{os.environ["HF_BASE"]}/multilegal_pile/filtering',
        #     # depends=stage3,
        # )
        #
        # stage4.run()

    finally:
        stop_event.set()
        resource_log_thread.join()

        # Log completion
        with open(log_file, "a") as f:
            f.write("All stages completed.\n")
