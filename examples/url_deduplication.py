from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup.url_dedup import (
    UrlDedupConfig,
    UrlDedupFilter,
    UrlDedupSignature,
    UrlFindDedups,
)
from datatrove.pipeline.dedup.sentence_dedup import SentDedupConfig
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.typeshelper import Languages


"""
example on how to use url-deduplication.

to run deduplication we need to run three different pipelines
"""

import argparse
import numpy as np


# modify url dedup hyper params here
url_dedup_config = UrlDedupConfig(
    document_priority=lambda doc: min(np.iinfo(np.uint16).max, len(doc.text) // 4),
    url_normalizer=lambda url: url.lower(),
)

FINDER_WORKERS = 4  # this will speed up/parallelize step 2


def run_example(args):
    pipeline_1 = [
        JsonlReader(args.input_folder),
        UrlDedupSignature(
            output_folder=f"{args.base_output_folder}/url-sigs",
            config=url_dedup_config,
            finder_workers=FINDER_WORKERS,
        ),
    ]

    pipeline_2 = [
        UrlFindDedups(
            data_folder=f"{args.base_output_folder}/url-sigs",
            output_folder=f"{args.base_output_folder}/url-dups",
            config=url_dedup_config,
        )
    ]

    pipeline_3 = [
        JsonlReader(data_folder="intermediate/"),
        UrlDedupFilter(data_folder="c4/dups", config=url_dedup_config),
    ]

    executor_1: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_1, workers=4, tasks=4
    )

    executor_2: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_2, workers=1, tasks=FINDER_WORKERS
    )

    executor_3: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_3, workers=4, tasks=4
    )

    print(executor_1.run())
    print(executor_2.run())
    print(executor_3.run())


parser = argparse.ArgumentParser(description="URL Deduplication")
parser.add_argument("input_folder", required=True, help="Input folder path")
parser.add_argument("base_output_folder", required=True, help="Base output folder path")
if __name__ == "__main__":
    args = parser.parse_args()
    run_example(args)
