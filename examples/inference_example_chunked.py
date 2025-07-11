"""
Chunked inference pipeline example with checkpointing.

This example shows how to run inference on documents using the InferenceRunner
with chunking enabled. Documents are processed in chunks with checkpoint support
for resuming from failures. Each chunk is saved to a separate output file.
"""

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceConfig
from datatrove.data import Document


def query_builder(document: Document) -> dict:
    """
    Query builder for vision-language model processing (e.g., OCR, image analysis).
    
    Args:
        document: Input document with image URL or content
        
    Returns:
        dict: Query payload for the inference server
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
OUTPUT_PATH = "./output_tmp"
CHECKPOINTS_PATH = "./checkpoints_tmp"  # Path for checkpoint files

documents = [
    Document(text="What's the weather in Tokyo?", id="0"),
    Document(text="What's the weather in France?", id="1"),
]



# Configure the inference settings with chunking
config = InferenceConfig(
    server_port=30024,
    server_type="lmdeploy",  # Options: "sglang", "vllm", "lmdeploy", "dummy"
    model_name_or_path="reducto/RolmOCR",
    model_chat_template="qwen2d5-vl",
    temperature=0.0,
    model_max_context=8192,
    max_concurrent_requests=100,
    max_concurrent_tasks=20,
    metric_interval=120,
    records_per_chunk=1,  # Enable chunking with 1000 documents per chunk
)

# Create the pipeline with chunking
pipeline_executor = LocalPipelineExecutor(
    pipeline=[
        # Read input documents
        documents,
        # Inference runner with chunking and checkpointing
        InferenceRunner(
            query_builder=query_builder,
            config=config,
            completions_dir=CHECKPOINTS_PATH,  # Enable checkpointing
            post_process_steps=[
                # Output writer with chunked filename pattern
                JsonlWriter(
                    OUTPUT_PATH,
                    output_filename="${rank}_chunk_${chunk_index}.jsonl",  # Chunked filename pattern
                )
            ]
        ),
    ],
    logging_dir="logs/inference_chunked",
    tasks=1,  # Number of parallel tasks
)

if __name__ == "__main__":
    print("🚀 Starting chunked inference pipeline...")
    print(f"📊 Configuration:")
    print(f"   - Output: {OUTPUT_PATH}")
    print(f"   - Checkpoints: {CHECKPOINTS_PATH}")
    print(f"   - Records per chunk: {config.records_per_chunk}")
    print(f"   - Server type: {config.server_type}")
    print(f"   - Model: {config.model_name_or_path}")
    
    # Run the pipeline
    pipeline_executor.run()
    
    print("✅ Chunked inference pipeline completed!")
    print(f"📁 Output files saved with pattern: [rank]_chunk_[chunk_index].jsonl")
    print(f"🔄 Checkpoint files saved to: {CHECKPOINTS_PATH}")
    print("💡 To resume from a checkpoint, simply re-run this script.") 