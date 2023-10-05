import os

from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import LocalInputDataFolder, LocalOutputDataFolder
from datatrove.pipeline.dedup import SentenceDedupFilter, SentenceDedupSignature, SentenceFindDedups
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.typeshelper import Languages


"""
example on how to use sentence-deduplication. sentence-deduplication implements deduplication as in:
https://jmlr.org/papers/v21/20-074.html
    'To deduplicate the data set, we discarded all but one of any three-sentence span
    occurring more than once in the data set.'

to run deduplication we need to run three different pipelines,
pipeline 1:
    implements usual extraction + quality filtering, it ends with SentenceDedupSignature, preprended by a writer.
pipeline 2:
    implements only SentenceFindDedups
pipeline 3:
    implements SentenceDedupFilter prepended by a reader of the same writer-kind used during stage 1. after the
    SentenceDedupFilter.
"""


def run_example():
    pipeline_1 = [
        WarcReader(data_folder=LocalInputDataFolder(path=f"{os.getcwd()}/warc/"), limit=1000),
        Trafilatura(),
        GopherQualityFilter(min_stop_words=0),
        LanguageFilter(language_threshold=0.5, languages=(Languages.english,)),
        JsonlWriter(LocalOutputDataFolder(path=f"{os.getcwd()}/intermediate/")),
        SentenceDedupSignature(output_folder=LocalOutputDataFolder(path=f"{os.getcwd()}/c4/")),
    ]

    pipeline_2 = [
        SentenceFindDedups(
            data_folder=LocalInputDataFolder(path=f"{os.getcwd()}/c4/"),
            output_folder=LocalOutputDataFolder(path=f"{os.getcwd()}/c4/"),
        )
    ]

    pipeline_3 = [
        JsonlReader(data_folder=LocalInputDataFolder(path=f"{os.getcwd()}/intermediate/")),
        SentenceDedupFilter(data_folder=LocalInputDataFolder(path=f"{os.getcwd()}/c4/")),
    ]

    executor_1: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_1, workers=4, max_concurrent_uploads=1, tasks=4
    )

    executor_2: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_2, workers=1, max_concurrent_uploads=1, tasks=1
    )

    executor_3: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_3, workers=4, max_concurrent_uploads=1, tasks=4
    )

    print(executor_1.run())
    print(executor_2.run())
    print(executor_3.run())


run_example()
