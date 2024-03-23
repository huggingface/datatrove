from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.stats import LanguageIdStats
from datatrove.pipeline.writers.jsonl import JsonlWriter


def run_step():
    pipeline_1 = [
        HuggingFaceDatasetReader(
            "truthful_qa", dataset_options={"name": "generation", "split": "validation"}, text_key="question"
        ),
        LanguageIdStats(),
        JsonlWriter("truthful_qa_with_language_id", compression=None),
    ]

    executor_1: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_1, workers=1, tasks=1)

    executor_1.run()
