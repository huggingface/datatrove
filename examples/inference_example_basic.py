"""
Basic inference pipeline example without chunking.

This example shows how to run inference on documents using the InferenceRunner
without chunking. Documents are processed and saved to a simple output structure.
"""

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceConfig
from datatrove.data import Document


def simple_query_builder(document: Document) -> dict:
    """
    Simple query builder that extracts text from document for OCR processing.
    
    Args:
        document: Input document with text content
        
    Returns:
        dict: Query payload for the inference server
    """
    return {
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Extract all text from this image."},
                    {"type": "image_url", "image_url": {"url": document.text}}
                ]
            }
        ],
        "max_tokens": 2048,
        "stream": False
    }


# Configuration
INPUT_PATH = "s3://your-bucket/input-documents"  # Path to input JSONL files
OUTPUT_PATH = "s3://your-bucket/inference-output"  # Path for output

# Configure the inference settings
config = InferenceConfig(
    server_port=30024,
    server_type="lmdeploy",  # Options: "sglang", "vllm", "lmdeploy", "dummy"
    model_name_or_path="reducto/RolmOCR",
    model_chat_template="qwen2d5-vl",
    temperature=0.0,
    model_max_context=8192,
    max_concurrent_requests=50,
    max_concurrent_tasks=10,
    metric_interval=120,
    # records_per_chunk is None for basic mode (no chunking)
)

# Create the pipeline
pipeline_executor = LocalPipelineExecutor(
    pipeline=[
        # Read input documents
        JsonlReader(INPUT_PATH),
        
        # Inference runner - processes documents through LLM
        InferenceRunner(
            query_builder=simple_query_builder,
            config=config,
            post_process_steps=[
                # Output writer - saves processed documents
                JsonlWriter(
                    OUTPUT_PATH,
                    output_filename="${rank}.jsonl"  # Simple filename pattern
                )
            ]
        ),
    ],
    logging_dir="logs/inference_basic"
)

if __name__ == "__main__":
    # Run the pipeline
    pipeline_executor.run()
    print("✅ Basic inference pipeline completed!") 