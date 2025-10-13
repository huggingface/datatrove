#!/usr/bin/env python3
"""
Example 3: Tokenization Pipeline

Tokenize documents using LLM tokenizers and analyze token statistics for ML workflows.

Components:
- JsonlReader: Read from HuggingFace C4 dataset
- TokensCounter: Count tokens with GPT-2 tokenizer
- LambdaFilter: Filter by token count (50-2048)
- JsonlWriter: Save tokenized results

Usage:
    python spec/phase1/examples/03_tokenization.py
    python spec/phase1/examples/03_tokenization.py analyze
    python spec/phase1/examples/03_tokenization.py compare
"""

import json
import os
import sys

from transformers import AutoTokenizer

from datatrove.utils.logging import logger
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter

# Configuration
OUTPUT_DIR = "spec/phase1/output/03_tokenized"
LOGS_DIR = "spec/phase1/logs/03_tokenization"


def main():
    """Main pipeline execution."""
    logger.info("Starting Example 3: Tokenization Pipeline")

    pipeline = [
        JsonlReader(
            "hf://datasets/allenai/c4/en/",
            glob_pattern="c4-train.00000-of-01024.json.gz",
            limit=1000,
        ),
        TokensCounter(
            tokenizer_name_or_path="gpt2",
            count_eos_token=True,
        ),
        LambdaFilter(
            lambda doc: 50 <= doc.metadata.get("token_count", 0) <= 2048,
        ),
        TokensCounter(
            tokenizer_name_or_path="gpt2",
            count_eos_token=True,
        ),
        JsonlWriter(
            output_folder=OUTPUT_DIR,
            output_filename="tokenized_${rank}.jsonl",
            compression=None
        )
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir=LOGS_DIR
    )

    executor.run()

    logger.info(f"Pipeline completed. Check: {OUTPUT_DIR}")


def analyze_tokenization():
    """Analyze the tokenization results."""
    output_file = f"{OUTPUT_DIR}/tokenized_00000.jsonl"

    if not os.path.exists(output_file):
        logger.warning("No output file found. Run the pipeline first.")
        return

    logger.info("Tokenization Analysis")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    with open(output_file, 'r') as f:
        docs = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Total documents after filtering: {len(docs)}")

    if docs:
        token_counts = [doc.get('metadata', {}).get('token_count', 0) for doc in docs]

        logger.info(f"Token count stats - Min: {min(token_counts)}, "
                   f"Max: {max(token_counts)}, "
                   f"Avg: {sum(token_counts) / len(token_counts):.0f}, "
                   f"Total: {sum(token_counts):,}")

        sample_doc = docs[0]
        sample_text = sample_doc.get('text', '')[:200]

        logger.info(f"Sample text: '{sample_text}...'")

        tokens = tokenizer.encode(sample_text)
        logger.info(f"Tokens ({len(tokens)}): {tokens[:20]}...")

        decoded_tokens = [tokenizer.decode([t]) for t in tokens[:10]]
        logger.info(f"First 10 decoded tokens: {decoded_tokens}")

        total_words = sum(len(doc.get('text', '').split()) for doc in docs)
        total_tokens = sum(token_counts)
        if total_words > 0:
            logger.info(f"Token/Word ratio: {total_tokens/total_words:.2f}")


def compare_tokenizers():
    """Compare different tokenizers on the same text."""
    sample_text = """
    DataTrove is a library to process, filter and deduplicate text data at a very large scale.
    It provides a set of prebuilt commonly used processing blocks with a framework to easily add custom functionality.
    """

    tokenizers = {
        "gpt2": AutoTokenizer.from_pretrained("gpt2"),
        "bert-base-uncased": AutoTokenizer.from_pretrained("bert-base-uncased"),
    }

    logger.info("Tokenizer Comparison")
    logger.info(f"Sample text ({len(sample_text)} chars): {sample_text}")

    for name, tok in tokenizers.items():
        tokens = tok.encode(sample_text)
        logger.info(f"{name}: {len(tokens)} tokens, "
                   f"First 10: {tokens[:10]}, "
                   f"Vocab size: {len(tok)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_tokenization()
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_tokenizers()
    else:
        main()
        analyze_tokenization()