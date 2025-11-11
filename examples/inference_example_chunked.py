"""
Chunked inference pipeline example with rollouts.

This example shows how to run inference on documents using the InferenceRunner
with checkpointing enabled. Documents are processed with a rollout function that
can perform multiple generations per document before the results are written.
"""

import asyncio
import atexit
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Awaitable, Callable

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner, InferenceResult
from datatrove.pipeline.writers import JsonlWriter


async def simple_rollout(
    runner: InferenceRunner,
    document: Document,
    generate: Callable[[dict[str, Any]], Awaitable[InferenceResult]],
) -> InferenceResult:
    """
    Basic rollout that sends a single request per document.

    Returns the InferenceResult directly, which will be stored under document.metadata["rollout_results"].
    """

    payload = {
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

    return await generate(payload)


async def chunked_rollout(
    runner: InferenceRunner,
    document: Document,
    generate: Callable[[dict[str, Any]], Awaitable[InferenceResult]],
) -> str:
    """
    Rollout that chunks long inputs and stitches the generations together.
    """

    instruction = "Rewrite this in a more formal style:"
    max_chars_per_part = 4000
    text = document.text
    chunks = [text[i : i + max_chars_per_part] for i in range(0, len(text), max_chars_per_part)] or [text]

    generations: list[dict[str, Any]] = []
    prev_chunk = None

    for chunk in chunks:
        # here we just ask the model to continue the previous generation or an empty msg if there isn't anything
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{instruction}\n\n{prev_chunk if prev_chunk else ''}{chunk}",
                },
                {
                    "role": "assistant",
                    "content": generations[-1] if generations else "",
                },
            ],
            # see https://docs.vllm.ai/en/v0.7.2/api/offline_inference/llm.html#vllm.LLM.chat
            "continue_final_message": True,
            "add_generation_prompt": False,
            "echo": False,
        }

        # could potentially have some error handling here
        result: InferenceResult = await generate(payload)
        generations.append(result.text)
        prev_chunk = chunk
    return "\n".join(generations)


def cpu_heavy_build_payload(doc: Document, page: int) -> dict[str, Any]:
    # simulate heavy work
    import time

    # not async on purpose
    time.sleep(10)
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"[page {page}] {doc.text}"}],
            }
        ],
        "max_tokens": 4096,
    }

async def heavy_cpu_rollout(
    runner: InferenceRunner,
    document: Document,
    generate: Callable[[dict[str, Any]], Awaitable[InferenceResult]],
) -> list[InferenceResult]:
    """
    Example rollout that offloads heavy preprocessing to a process pool.
    """

    if not hasattr(runner, "process_pool"):
        runner.process_pool = ProcessPoolExecutor(max_workers=100)
        atexit.register(runner.process_pool.shutdown, wait=False, cancel_futures=True)

    loop = asyncio.get_running_loop()

    async def process_page(page: int) -> InferenceResult:
        payload = await loop.run_in_executor(runner.process_pool, cpu_heavy_build_payload, document, page)
        return await generate(payload)

    page_results = await asyncio.gather(*[process_page(page) for page in [1, 2]], return_exceptions=True)

    return page_results


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
    model_max_context=8192,
    metric_interval=120,
    default_generation_params={"temperature": 0.0},
    rollouts_per_document=1,
    max_concurrent_generations=500,
)

# Create the pipeline with chunking
pipeline_executor: LocalPipelineExecutor = LocalPipelineExecutor(
    pipeline=[
        documents,
        InferenceRunner(
            rollout_fn=chunked_rollout,
            config=config,
            records_per_chunk=500,  # Enable chunking with 500 documents per chunk
            checkpoints_local_dir=CHECKPOINTS_PATH,  # Leave unset to disable checkpointing
            output_writer=JsonlWriter(OUTPUT_PATH, output_filename="${rank}_chunk_${chunk_index}.jsonl"),
        ),
    ],
    logging_dir=LOGS_PATH,
    tasks=1,  # Number of parallel tasks
)

if __name__ == "__main__":
    # Run the pipeline
    pipeline_executor.run()
