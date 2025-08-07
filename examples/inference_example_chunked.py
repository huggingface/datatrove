"""
Chunked inference pipeline example with chunking.

This example shows how to run inference on documents using the InferenceRunner
with chunking enabled. Documents are processed in chunks with checkpoint support
for resuming from failures. Each chunk is saved to a separate output file.
"""

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.writers import JsonlWriter
from typing import Any


"""
You can use either an async query builder yielding queries or siple sync query builder, which just yields a single query.
"""

def query_builder(runner: InferenceRunner, document: Document) -> dict[str, Any]:
    """
    Query builder for Language Model.
    
    Args:
        document: Input document with image URL or content
        
    Returns:
        Query payload dictionary for the inference server containing messages and max_tokens
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
        "max_tokens": 4096,
    }

# Configuration
OUTPUT_PATH: str = "s3://.../final_output_data"
LOGS_PATH: str = "/fsx/.../finetranslations/inference_logs"
CHECKPOINTS_PATH: str = "/fsx/.../finetranslations/translate-checkpoints"  # Path for checkpoint files

# 1000 documents
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
            query_builder=query_builder,
            config=config,
            records_per_chunk=500,  # Enable chunking with 500 documents per chunk
            checkpoints_local_dir=CHECKPOINTS_PATH,
            output_writer=JsonlWriter(OUTPUT_PATH, output_filename="${rank}_chunk_${chunk_index}.jsonl"),
        ),
    ],
    logging_dir=LOGS_PATH,
    tasks=1,  # Number of parallel tasks
)

if __name__ == "__main__":
    # Run the pipeline
    pipeline_executor.run() 