"""
Example 1: Basic Data Processing Pipeline
==========================================
Learn the fundamentals of DataTrove by reading from HuggingFace C4 dataset,
applying filters, and saving results.

This example demonstrates:
- Reading from HuggingFace datasets directly
- Using LambdaFilter for custom filtering
- Pipeline composition
- Local execution
"""

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


def main():
    # Create the processing pipeline
    pipeline = [
        # Read from HuggingFace C4 dataset
        # Using just one shard for quick local testing
        JsonlReader(
            "hf://datasets/allenai/c4/en/",
            glob_pattern="c4-train.00000-of-01024.json.gz",  # Just first shard
            limit=1000,  # Only read 1000 documents for testing
        ),

        # Filter 1: Keep documents with sufficient length
        LambdaFilter(lambda doc: len(doc.text) > 100),

        # Filter 2: Keep documents containing certain keywords (case-insensitive)
        LambdaFilter(
            lambda doc: any(keyword in doc.text.lower()
                          for keyword in ["data", "learning", "computer", "science"])
        ),

        # Filter 3: Random sampling (keep 50% of remaining documents)
        SamplerFilter(rate=0.5),

        # Save the filtered results
        JsonlWriter(
            output_folder="spec/phase1/output/01_filtered",
            output_filename="filtered_${rank}.jsonl",
            compression=None  # No compression for easier inspection
        )
    ]

    # Create and run the executor
    print("Starting Example 1: Basic Filtering Pipeline")
    print("=" * 50)

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,  # Start with single task
        logging_dir="spec/phase1/logs/01_basic_filtering"
    )

    # Run the pipeline
    executor.run()

    print("\n" + "=" * 50)
    print("Pipeline completed!")
    print("Check outputs in: spec/phase1/output/01_filtered/")
    print("Check logs in: spec/phase1/logs/01_basic_filtering/")


def inspect_results():
    """Helper function to inspect the results"""
    import json
    import os

    output_file = "spec/phase1/output/01_filtered/filtered_00000.jsonl"

    if not os.path.exists(output_file):
        print("No output file found. Run the pipeline first.")
        return

    print("\n" + "=" * 50)
    print("Inspecting Results")
    print("=" * 50)

    # Count documents
    with open(output_file, 'r') as f:
        docs = [json.loads(line) for line in f]

    print(f"Total documents after filtering: {len(docs)}")

    if docs:
        # Show sample document
        print("\nSample document:")
        print(f"ID: {docs[0].get('id', 'N/A')}")
        print(f"Text length: {len(docs[0].get('text', ''))}")
        print(f"Text preview: {docs[0].get('text', '')[:200]}...")

        # Basic statistics
        text_lengths = [len(doc.get('text', '')) for doc in docs]
        print(f"\nText length statistics:")
        print(f"  Min: {min(text_lengths)}")
        print(f"  Max: {max(text_lengths)}")
        print(f"  Avg: {sum(text_lengths) / len(text_lengths):.0f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        inspect_results()
    else:
        main()
        # Automatically inspect results after running
        inspect_results()