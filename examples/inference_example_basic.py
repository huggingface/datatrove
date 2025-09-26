"""
Basic inference pipeline example without chunking.

This example shows how to run inference on documents using the InferenceRunner
without chunking. Documents are processed and saved to a simple output structure.
"""

import asyncio
import atexit
from typing import Any
from typing import AsyncGenerator
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceSuccess, InferenceError
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable


# For creating query payloads, you have 2 options:
# 1. Create a simple query builder that returns a dict
def simple_query_builder(runner: InferenceRunner, document: Document) -> dict[str, Any]:
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
                ]
            }
        ],
        "max_tokens": 2048,
    }

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


    for page in [1,2]:
        yield await asyncio.get_running_loop().run_in_executor(runner.process_pool, heavy_cpu_task, document, page)

class PostProcessInferenceResults(PipelineStep):
    """
    Post-process inference results.
    """

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            document.text = "\n".join([x.text if isinstance(x, InferenceSuccess) else x.error for x in document.metadata["inference_results"]])
            del document.metadata["inference_results"]
            yield document


documents = [Document(text=f"What's {i} + {i}? Then generate letter 'd' this many times", id=str(i)) for i in range(1000)]

# Configuration
OUTPUT_PATH = "./output_tmp"  # Path for output

# Configure the inference settings
config = InferenceConfig(
    server_type="vllm",
    model_name_or_path="reducto/RolmOCR",
    temperature=0.0,
    model_max_context=8192,
    max_concurrent_requests=500,
    max_concurrent_tasks=1000,
    # Report metrics every 2 minutes
    metric_interval=5,
)

# Create the pipeline
pipeline_executor = LocalPipelineExecutor(
    pipeline=[
        documents,
        # Inference runner - processes documents through LLM
        InferenceRunner(
            query_builder=async_query_builder,
            config=config,
            post_process_steps=[
                PostProcessInferenceResults(),
                JsonlWriter(
                    OUTPUT_PATH,
                )
            ]
        ),
    ],
    logging_dir=None,
)

if __name__ == "__main__":
    # Run the pipeline
    pipeline_executor.run()