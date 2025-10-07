"""
Example 3: Tokenization Pipeline
=================================
Tokenize documents using LLM tokenizers and analyze token statistics.

This example demonstrates:
- Tokenizing text with GPT-2 tokenizer
- Counting tokens before and after filtering
- Token-based filtering
- Understanding tokenization for ML workflows
"""

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter


def main():
    # Create the processing pipeline
    pipeline = [
        # Read from HuggingFace C4 dataset
        JsonlReader(
            "hf://datasets/allenai/c4/en/",
            glob_pattern="c4-train.00000-of-01024.json.gz",
            limit=1000,  # Process 1000 documents
        ),

        # Count tokens BEFORE filtering
        TokensCounter(
            tokenizer_name_or_path="gpt2",  # Use GPT-2 tokenizer
            count_eos_token=True,    # Include EOS token in count
        ),

        # Filter by token count - keep documents with reasonable token counts
        # (not too short, not too long for typical LLM context windows)
        LambdaFilter(
            lambda doc: 50 <= doc.metadata.get("token_count", 0) <= 2048,
        ),

        # Count tokens AFTER filtering to see the difference
        TokensCounter(
            tokenizer_name_or_path="gpt2",
            count_eos_token=True,
        ),

        # Save the results
        JsonlWriter(
            output_folder="spec/phase1/output/03_tokenized",
            output_filename="tokenized_${rank}.jsonl",
            compression=None  # No compression for easier inspection
        )
    ]

    print("Starting Example 3: Tokenization Pipeline")
    print("=" * 50)

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir="spec/phase1/logs/03_tokenization"
    )

    executor.run()

    print("\n" + "=" * 50)
    print("Pipeline completed!")
    print("Check output in: spec/phase1/output/03_tokenized/")


def analyze_tokenization():
    """Analyze the tokenization results"""
    import json
    import os
    from transformers import AutoTokenizer

    output_file = "spec/phase1/output/03_tokenized/tokenized_00000.jsonl"

    if not os.path.exists(output_file):
        print("No output file found. Run the pipeline first.")
        return

    print("\n" + "=" * 50)
    print("Tokenization Analysis")
    print("=" * 50)

    # Load tokenizer for demonstration
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    with open(output_file, 'r') as f:
        docs = [json.loads(line) for line in f if line.strip()]

    print(f"\nTotal documents after filtering: {len(docs)}")

    if docs:
        # Token statistics
        token_counts = [doc.get('metadata', {}).get('token_count', 0) for doc in docs]

        print(f"\nToken count statistics:")
        print(f"  Min: {min(token_counts)} tokens")
        print(f"  Max: {max(token_counts)} tokens")
        print(f"  Average: {sum(token_counts) / len(token_counts):.0f} tokens")
        print(f"  Total: {sum(token_counts):,} tokens")

        # Show a sample tokenization
        sample_doc = docs[0]
        sample_text = sample_doc.get('text', '')[:200]  # First 200 chars

        print(f"\nSample tokenization:")
        print(f"Text: '{sample_text}...'")

        tokens = tokenizer.encode(sample_text)
        print(f"Tokens ({len(tokens)}): {tokens[:20]}...")  # First 20 tokens

        decoded_tokens = [tokenizer.decode([t]) for t in tokens[:10]]
        print(f"First 10 decoded tokens: {decoded_tokens}")

        # Token/word ratio
        total_words = sum(len(doc.get('text', '').split()) for doc in docs)
        total_tokens = sum(token_counts)
        if total_words > 0:
            print(f"\nToken/Word ratio: {total_tokens/total_words:.2f}")


def compare_tokenizers():
    """Compare different tokenizers on the same text"""
    from transformers import AutoTokenizer

    sample_text = """
    DataTrove is a library to process, filter and deduplicate text data at a very large scale.
    It provides a set of prebuilt commonly used processing blocks with a framework to easily add custom functionality.
    """

    tokenizers = {
        "gpt2": AutoTokenizer.from_pretrained("gpt2"),
        "bert-base-uncased": AutoTokenizer.from_pretrained("bert-base-uncased"),
    }

    print("\n" + "=" * 50)
    print("Tokenizer Comparison")
    print("=" * 50)
    print(f"Sample text ({len(sample_text)} chars):")
    print(sample_text)

    for name, tok in tokenizers.items():
        tokens = tok.encode(sample_text)
        print(f"\n{name}:")
        print(f"  Token count: {len(tokens)}")
        print(f"  First 10 tokens: {tokens[:10]}")
        print(f"  Vocab size: {len(tok)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_tokenization()
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_tokenizers()
    else:
        main()
        analyze_tokenization()