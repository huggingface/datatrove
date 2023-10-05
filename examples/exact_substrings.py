import os

from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import LocalInputDataFolder, LocalOutputDataFolder
from datatrove.pipeline.dedup import DatasetToSequence, DedupReader, MergeSequences
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.typeshelper import Languages


"""
example on how to run exact-substring deduplication. It also requires using
https://github.com/google-research/deduplicate-text-datasets after stage 1, 2
1) DatasetToSequence maps 1 file into a sequence S. With unique separators at the beginning of each document. It also
    saves the bytes offset of where each individual document begins.
2) MergeSequences merges all sequences into a big single sequence. It also saves the bytes offset per file.

---
after stage two you should use deduplicate-text-datasets scripts to create the suffix array and find all the
duplicates. The final output of these scripts should be a .bytearange file with the ranges in bytes wrt the big
sequence
---

3) DedupReader reads from DocumentsPipeline and duplicates ranges at the same time removing the duplicates ranges.


to run stage 1,2 call run_stage_1_2, after you have followed deduplicate-text-datasets instructions in the README you
can call stage 3 with run_stage_3.

N.B
The steps

"""


def run_stage_1_2():
    pipeline_1 = [
        WarcReader(data_folder=LocalInputDataFolder(path=f"{os.getcwd()}/warc/"), limit=1000),
        Trafilatura(),
        GopherQualityFilter(min_stop_words=0),
        LanguageFilter(language_threshold=0.5, languages=(Languages.english,)),
        JsonlWriter(LocalOutputDataFolder(path=f"{os.getcwd()}/intermediate/")),
        DatasetToSequence(output_folder=LocalOutputDataFolder(path=f"{os.getcwd()}/es/")),
    ]

    pipeline_2 = [
        MergeSequences(
            input_folder=LocalInputDataFolder(path=f"{os.getcwd()}/es"),
            output_folder=LocalOutputDataFolder(path=f"{os.getcwd()}/es/"),
            tasks_stage_1=4,
        )
    ]

    executor_1: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_1, workers=4, max_concurrent_uploads=1, tasks=4
    )

    executor_2: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_2, workers=1, max_concurrent_uploads=1, tasks=1
    )

    print(executor_1.run())
    print(executor_2.run())


def run_stage_3():
    pipeline_3 = [
        DedupReader(
            LocalInputDataFolder(path=f"{os.getcwd()}/intermediate/"),
            sequence_folder=LocalInputDataFolder(path=f"{os.getcwd()}/es/"),
            test=False,
        )
    ]

    executor_3: PipelineExecutor = LocalPipelineExecutor(
        pipeline=pipeline_3, workers=4, max_concurrent_uploads=1, tasks=4
    )

    print(executor_3.run())
