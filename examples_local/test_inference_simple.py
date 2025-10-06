#!/usr/bin/env python3
"""
Test inference_example_basic pattern with just 3 documents and simple query builder.

This tests whether JsonlWriter correctly saves all documents when used in
InferenceRunner's post_process_steps with multiple documents.
"""
import sys
sys.path.insert(0, 'src')

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceSuccess, InferenceError
from typing import Iterable


class PostProcessInferenceResults(PipelineStep):
    """Post-process inference results."""

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            document.text = "\n".join([x.text if isinstance(x, InferenceSuccess) else x.error for x in document.metadata["inference_results"]])
            del document.metadata["inference_results"]
            yield document


def simple_query_builder(runner, document):
    """Simple query builder that just asks a question."""
    return {
        "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
        "max_tokens": 50,
    }


def main():
    # Just 3 documents for quick testing
    documents = [Document(text=f"What is {i} + {i}?", id=f"doc_{i}") for i in [1, 2, 3]]

    config = InferenceConfig(
        server_type="lmdeploy",
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.0,
        model_max_context=2048,
        max_concurrent_requests=1,
        max_concurrent_tasks=1,
    )

    pipeline_executor = LocalPipelineExecutor(
        pipeline=[
            documents,
            InferenceRunner(
                query_builder=simple_query_builder,
                config=config,
                post_process_steps=[
                    PostProcessInferenceResults(),
                    JsonlWriter("examples_local/output/inference_test")
                ]
            ),
        ],
        logging_dir=None,
    )

    print(f"Testing with {len(documents)} documents...")
    pipeline_executor.run()
    print("\nChecking output...")

    import subprocess
    result = subprocess.run(
        "zcat examples_local/output/inference_test/00000.jsonl.gz | wc -l",
        shell=True,
        capture_output=True,
        text=True
    )
    print(f"Documents written: {result.stdout.strip()}")

    result = subprocess.run(
        "zcat examples_local/output/inference_test/00000.jsonl.gz | jq -r '.id'",
        shell=True,
        capture_output=True,
        text=True
    )
    print(f"Document IDs:\n{result.stdout}")


if __name__ == "__main__":
    main()