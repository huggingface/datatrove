"""
Basic inference pipeline example without chunking.

This example shows how to run inference on documents using the InferenceRunner
without chunking. Documents are processed and saved to a simple output structure.
"""

from typing import Any

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceSuccess, InferenceError
from typing import Iterable


def simple_query_builder(document: Document) -> dict[str, Any]:
    """
    Simple query builder that extracts text from document for OCR processing.
    
    Args:
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
    server_type="sglang",
    model_name_or_path="reducto/RolmOCR",
    temperature=0.0,
    model_max_context=8192,
    max_concurrent_requests=500,
    max_concurrent_tasks=1000,
    # Report metrics every 2 minutes
    metric_interval=120,
)

# Create the pipeline
pipeline_executor = LocalPipelineExecutor(
    pipeline=[
        documents,
        # Inference runner - processes documents through LLM
        InferenceRunner(
            query_builder=simple_query_builder,
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