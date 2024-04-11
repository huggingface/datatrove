import argparse

import numpy as np

from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup.url_dedup import (
    UrlDedupConfig,
    UrlDedupFilter,
    UrlDedupSignature,
    UrlFindDedups,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


"""
Example on how to use url-deduplication.
To run url deduplication we need to run three different pipelines (same as sentence dedup)
"""


# modify url dedup hyper params here
url_dedup_config = UrlDedupConfig(
    # this will keep the longest document for each url
    document_priority=lambda doc: min(np.iinfo(np.uint16).max, len(doc.text) // 4),
    url_normalizer=lambda url: url.lower(),
)

FINDER_WORKERS = 4  # this will speed up/parallelize step 2

LIMIT = -1  # for testing


def run_example(args):
    pipeline_1 = [
        JsonlReader(args.input_folder, limit=LIMIT, progress=True),
        UrlDedupSignature(
            output_folder=f"{args.sigs_dup_folder}/sigs",
            config=url_dedup_config,
            finder_workers=FINDER_WORKERS,
        ),
    ]

    pipeline_2 = [
        UrlFindDedups(
            data_folder=f"{args.sigs_dup_folder}/sigs",
            output_folder=f"{args.sigs_dup_folder}/dups",
            config=url_dedup_config,
        )
    ]

    pipeline_3 = [
        JsonlReader(data_folder=args.input_folder, limit=LIMIT, progress=True),
        UrlDedupFilter(
            data_folder=f"{args.sigs_dup_folder}/dups",
            config=url_dedup_config,
            exclusion_writer=JsonlWriter(output_folder=f"{args.base_output_folder}/removed"),
        ),
        JsonlWriter(output_folder=f"{args.base_output_folder}/output"),
    ]

    executor_1: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_1, tasks=4)

    executor_2: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_2, tasks=FINDER_WORKERS)

    executor_3: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_3, tasks=4)

    print(executor_1.run())
    print(executor_2.run())
    print(executor_3.run())


parser = argparse.ArgumentParser(description="URL Deduplication")
parser.add_argument("input_folder", help="Input folder path")
parser.add_argument("base_output_folder", help="Base output folder path")
parser.add_argument("sigs_dup_folder", help="sigs-dup folder path")
if __name__ == "__main__":
    args = parser.parse_args()
    run_example(args)
