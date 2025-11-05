"""
Chunked inference pipeline example with chunking.

This example shows how to run inference on documents using the InferenceRunner
with chunking enabled. Documents are processed in chunks with checkpoint support
for resuming from failures. Each chunk is saved to a separate output file.
"""

from typing import Any, AsyncGenerator

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.writers import JsonlWriter


# For creating query payloads, you have 2 options:
# 1. Create a simple query builder that returns a dict
def simple_query_builder(runner: InferenceRunner, document: Document) -> dict[str, Any] | None:
    """
    Simple query builder that extracts text from document for OCR processing.

    Args:
        runner: Inference runner instance
        document: Input document with text content

    Returns:
        Query payload for the inference server
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": document.text},
                ],
            }
        ],
        "max_tokens": 2048,
    }


def large_sample_query_builder(runner: InferenceRunner, document: Document) -> dict[str, Any] | None:
    """Query builder that chunks long samples and requests callbacks for continuation."""

    MAX_CHARS_PER_PART = 4000
    instruction = "Rewrite this in a more formal style:"
    chunks = document.metadata.get("chunks")
    if not chunks:
        text = document.text
        if len(text) > MAX_CHARS_PER_PART:
            chunks = [text[i : i + MAX_CHARS_PER_PART] for i in range(0, len(text), MAX_CHARS_PER_PART)]
            document.metadata["chunks"] = chunks
        else:
            chunks = [text]

    inference_results = document.metadata.get("inference_results") or []
    total_parts = len(chunks)
    current_index = min(len(inference_results), total_parts - 1)
    current_chunk = chunks[current_index]

    if current_index == 0:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{instruction}\n\n{current_chunk}",
                }
            ],
        }
    else:
        previous_chunk = chunks[current_index - 1]
        previous_result = inference_results[-1]
        previous_generation = getattr(previous_result, "text", str(previous_result))
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{instruction}\n\n{previous_chunk}{current_chunk}",
                },
                {
                    "role": "assistant",
                    "content": previous_generation,
                },
            ],
            # see these params here https://docs.vllm.ai/en/v0.7.2/api/offline_inference/llm.html#vllm.LLM.chat
            "continue_final_message": True,
            "add_generation_prompt": False,
            "echo": False,
        }

    # if we have a bunch of chunks for this sample, we want this function to be called again after the next generation is completed
    payload["callback"] = len(inference_results) < total_parts - 1
    return payload


# 2. Create an async query builder that returns an async generator of dicts. Use this option if you need
# a) Create multiple requests per document
# b) Your query function is IO/CPU heavy


def heavy_cpu_task(document: Document, page: int):
    # block sleep
    import time

    time.sleep(10)
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": document.text}],
            }
        ],
        "max_tokens": 4096,
    }


async def async_query_builder(runner: InferenceRunner, document: Document) -> AsyncGenerator[dict[str, Any], None]:
    """
    Query builder for Language Model.

    Args:
        document: Input document with image URL or content

    Returns:
        Async generator of query payloads for the inference server
    """
    import asyncio
    import atexit
    from concurrent.futures import ProcessPoolExecutor

    # Because it's async, you can run IO heavy tasks with little to no overhead (simply use await)
    # If you need to run CPU heavy tasks, it's a bit more complicated
    # 1. create a process pool executor and bind it to the runner
    # 2. access the process pool, then using asyncio.run_in_executor

    # If we didn't run with this the whole execution would take at least 1000*2*10 seconds
    if not hasattr(runner, "process_pool"):
        runner.process_pool = ProcessPoolExecutor(max_workers=100)
        runner.process_pool.__enter__()
        # Register cleanup
        atexit.register(runner.process_pool.__exit__, None, None, None)

    for page in [1, 2]:
        yield await asyncio.get_running_loop().run_in_executor(runner.process_pool, heavy_cpu_task, document, page)


# Configuration
OUTPUT_PATH: str = "s3://.../final_output_data"
LOGS_PATH: str = "/fsx/.../finetranslations/inference_logs"
CHECKPOINTS_PATH: str = "/fsx/.../finetranslations/translate-checkpoints"  # Path for checkpoint files

# 1005 documents
documents = [Document(text="What's the weather in Tokyo?", id=str(i)) for i in range(1005)]


# Configure the inference settings with chunking
config: InferenceConfig = InferenceConfig(
    server_type="vllm",  # Options: "sglang", "vllm", "dummy"
    model_name_or_path="reducto/RolmOCR",
    temperature=0.0,
    model_max_context=8192,
    max_concurrent_requests=500,
    max_concurrent_tasks=500,
    metric_interval=120,
)

# Create the pipeline with chunking
pipeline_executor: LocalPipelineExecutor = LocalPipelineExecutor(
    pipeline=[
        # Read input documents
        documents,
        InferenceRunner(
            query_builder=large_sample_query_builder,
            config=config,
            records_per_chunk=500,  # Enable chunking with 500 documents per chunk
            checkpoints_local_dir=CHECKPOINTS_PATH,  # leave unset to disable checkpointing behaviour
            output_writer=JsonlWriter(OUTPUT_PATH, output_filename="${rank}_chunk_${chunk_index}.jsonl"),
            # you can also pass a postprocess_fn(document) -> document|None to modify/filter the document after inference. Return None to remove the document
            postprocess_fn=None,
        ),
    ],
    logging_dir=LOGS_PATH,
    tasks=1,  # Number of parallel tasks
)

if __name__ == "__main__":
    # Run the pipeline
    pipeline_executor.run()
