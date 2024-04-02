from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import SentenceDedupFilter, SentenceDedupSignature, SentenceFindDedups
from datatrove.pipeline.dedup.sentence_dedup import SentDedupConfig
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

# modify sentence dedup hyper params here
sent_dedup_config = SentDedupConfig(
    n_sentences=3,
    split_sentences=True,  # set to False to split on \n instead
    only_dedup_in_index=True,
    min_doc_words=50,
)

FINDER_WORKERS = 10  # this will speed up/parallelize step 2


def run_example():
    pipeline_1 = [
        WarcReader(data_folder="warc/", limit=1000),
        Trafilatura(),
        GopherQualityFilter(min_stop_words=0),
        LanguageFilter(language_threshold=0.5, languages=(Languages.english,)),
        JsonlWriter("intermediate/"),
        SentenceDedupSignature(output_folder="c4/sigs", config=sent_dedup_config, finder_workers=FINDER_WORKERS),
    ]

    pipeline_2 = [SentenceFindDedups(data_folder="c4/sigs", output_folder="c4/dups", config=sent_dedup_config)]

    pipeline_3 = [
        JsonlReader(data_folder="intermediate/"),
        SentenceDedupFilter(data_folder="c4/dups", config=sent_dedup_config),
    ]

    executor_1: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_1, workers=4, tasks=4)

    executor_2: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_2, workers=1, tasks=FINDER_WORKERS)

    executor_3: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_3, workers=4, tasks=4)

    print(executor_1.run())
    print(executor_2.run())
    print(executor_3.run())


if __name__ == "__main__":
    run_example()
