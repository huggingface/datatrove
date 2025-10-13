#!/usr/bin/env python3
"""
Example 5: Deduplication Pipeline

Remove duplicate and near-duplicate content using hash-based exact deduplication.

Components:
- JsonlReader: Read documents from synthetic test data
- LambdaFilter: Hash-based exact duplicate detection and removal
- JsonlWriter: Save deduplicated results

Usage:
    python spec/phase1/examples/05_deduplication.py
    python spec/phase1/examples/05_deduplication.py analyze
    python spec/phase1/examples/05_deduplication.py c4
"""

import hashlib
import json
import os
import shutil
import sys

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import (
    MinhashConfig,
    MinhashDedupFilter,
    MinhashDedupSignature,
    SingleBloomFilter,
)
from datatrove.pipeline.dedup.bloom_filter import BloomFilterConfig
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.logging import logger

# Configuration
OUTPUT_DIR = "spec/phase1/output/05_dedup_hash"
LOGS_DIR = "spec/phase1/logs/05_dedup_hash"


def create_sample_data_with_duplicates():
    """Create sample data with intentional duplicates for testing"""

    documents = []

    # Add some unique documents
    unique_texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Python is a versatile programming language widely used in data science and web development.",
        "Neural networks are inspired by biological neural networks in animal brains.",
        "Data preprocessing is crucial for building effective machine learning models.",
        "Deep learning has revolutionized computer vision and natural language processing.",
    ]

    for i, text in enumerate(unique_texts):
        documents.append({
            "text": text,
            "id": f"unique_{i:03d}",
            "metadata": {"source": "original"}
        })

    # Add exact duplicates
    for i in range(3):
        documents.append({
            "text": unique_texts[0],  # Duplicate of first text
            "id": f"exact_dup_{i:03d}",
            "metadata": {"source": "exact_duplicate"}
        })

    # Add near duplicates (slight variations)
    near_dup_base = unique_texts[1]
    variations = [
        near_dup_base.replace("Python", "Python3"),
        near_dup_base.replace(".", "!"),
        near_dup_base + " It has simple syntax.",
        "Python is a versatile programming language. It's widely used in data science and web development.",
    ]

    for i, text in enumerate(variations):
        documents.append({
            "text": text,
            "id": f"near_dup_{i:03d}",
            "metadata": {"source": "near_duplicate"}
        })

    # Add some C4 documents for variety
    documents.extend([
        {
            "text": "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the alphabet.",
            "id": "pangram_001",
            "metadata": {"source": "misc"}
        },
        {
            "text": "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the alphabet.",
            "id": "pangram_002",
            "metadata": {"source": "exact_duplicate"}
        }
    ])

    # Save to file
    output_file = "spec/phase1/data/sample_with_duplicates.jsonl"
    with open(output_file, 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')

    logger.info(f"Created {len(documents)} documents with duplicates: "
                f"{len(unique_texts)} unique, 3 exact duplicates, "
                f"{len(variations)} near duplicates, 2 pangram documents (1 duplicate)")
    logger.info(f"Saved to: {output_file}")

    return len(documents)


def exact_dedup_pipeline():
    """Run exact deduplication using Bloom filter"""

    logger.info("Running Exact Deduplication with Bloom Filter")

    # Bloom filter config for exact matching
    bloom_config = BloomFilterConfig(
        expected_elements=1000,  # Expected number of unique documents
        false_positive_rate=0.001,  # Low false positive rate
        hash_func="sha256",  # Hash function for document fingerprints
    )

    pipeline = [
        JsonlReader(
            data_folder="spec/phase1/data",
            glob_pattern="sample_with_duplicates.jsonl",
        ),

        # Bloom filter for exact deduplication
        SingleBloomFilter(
            output_folder="spec/phase1/output/05_dedup_bloom",
            config=bloom_config,
            save_bloom_filter=True,
        ),

        JsonlWriter(
            output_folder="spec/phase1/output/05_dedup_exact",
            compression=None,
        )
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir="spec/phase1/logs/05_dedup_exact"
    )

    executor.run()


def minhash_dedup_pipeline():
    """Run near-duplicate detection using MinHash"""

    logger.info("Running Near-Duplicate Detection with MinHash")

    # First, compute MinHash signatures
    signature_pipeline = [
        JsonlReader(
            data_folder="spec/phase1/data",
            glob_pattern="sample_with_duplicates.jsonl",
        ),

        MinhashDedupSignature(
            output_folder="spec/phase1/output/05_minhash_sigs",
            config=MinhashConfig(
                num_buckets=10,  # Number of hash buckets
                hashes_per_bucket=10,  # Hashes per bucket
            ),
        ),
    ]

    logger.info("Step 1: Computing MinHash signatures...")
    sig_executor = LocalPipelineExecutor(
        pipeline=signature_pipeline,
        tasks=1,
        logging_dir="spec/phase1/logs/05_minhash_sig"
    )
    sig_executor.run()

    # Then filter based on signatures
    filter_pipeline = [
        JsonlReader(
            data_folder="spec/phase1/data",
            glob_pattern="sample_with_duplicates.jsonl",
        ),

        MinhashDedupFilter(
            input_folder="spec/phase1/output/05_minhash_sigs",
            config=MinhashConfig(
                num_buckets=10,
                hashes_per_bucket=10,
            ),
        ),

        JsonlWriter(
            output_folder="spec/phase1/output/05_dedup_minhash",
            compression=None,
        )
    ]

    logger.info("Step 2: Filtering duplicates based on signatures...")
    filter_executor = LocalPipelineExecutor(
        pipeline=filter_pipeline,
        tasks=1,
        logging_dir="spec/phase1/logs/05_minhash_filter"
    )
    filter_executor.run()


def hash_based_dedup():
    """Simple hash-based exact deduplication"""

    logger.info("Running Simple Hash-Based Deduplication")

    seen_hashes = set()

    def is_duplicate(doc):
        # Create hash of document text
        doc_hash = hashlib.sha256(doc.text.encode()).hexdigest()
        if doc_hash in seen_hashes:
            return False  # It's a duplicate, filter it out
        seen_hashes.add(doc_hash)
        return True  # It's unique, keep it

    pipeline = [
        JsonlReader(
            data_folder="spec/phase1/data",
            glob_pattern="sample_with_duplicates.jsonl",
        ),

        # Filter duplicates using hash
        LambdaFilter(is_duplicate),

        JsonlWriter(
            output_folder=OUTPUT_DIR,
            compression=None,
        )
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir=LOGS_DIR
    )

    executor.run()

    logger.info(f"Unique documents found: {len(seen_hashes)}")


def analyze_results():
    """Analyze deduplication results"""

    logger.info("Deduplication Results Analysis")

    # Count original documents
    original_file = "spec/phase1/data/sample_with_duplicates.jsonl"
    with open(original_file, 'r') as f:
        original_count = sum(1 for _ in f)

    logger.info(f"Original documents: {original_count}")

    # Check results from different methods
    results = {
        "Hash-based": "spec/phase1/output/05_dedup_hash/00000.jsonl",
        "Bloom filter": "spec/phase1/output/05_dedup_exact/00000.jsonl",
        "MinHash": "spec/phase1/output/05_dedup_minhash/00000.jsonl",
    }

    for method, filepath in results.items():
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                lines = f.readlines()
                count = len(lines)
                removed = original_count - count
                pct = removed / original_count * 100
                logger.info(f"{method} - Remaining: {count}, Removed: {removed} ({pct:.1f}%)")

                # Show sample of what was kept
                if lines:
                    doc = json.loads(lines[0])
                    logger.info(f"  Sample kept: '{doc['text'][:50]}...'")


def dedup_c4_data():
    """Apply deduplication to real C4 data"""

    logger.info("Deduplicating C4 Data")

    seen_hashes = set()

    def is_duplicate(doc):
        doc_hash = hashlib.sha256(doc.text.encode()).hexdigest()
        if doc_hash in seen_hashes:
            return False
        seen_hashes.add(doc_hash)
        return True

    pipeline = [
        JsonlReader(
            "hf://datasets/allenai/c4/en/",
            glob_pattern="c4-train.00000-of-01024.json.gz",
            limit=5000,  # Process 5000 documents
        ),

        # Filter duplicates
        LambdaFilter(is_duplicate),

        JsonlWriter(
            output_folder="spec/phase1/output/05_c4_dedup",
            compression=None,
        )
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir="spec/phase1/logs/05_c4_dedup"
    )

    executor.run()

    logger.info(f"Unique documents in C4 sample: {len(seen_hashes)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_results()
    elif len(sys.argv) > 1 and sys.argv[1] == "c4":
        # Clean C4 output
        if os.path.exists("spec/phase1/output/05_c4_dedup"):
            shutil.rmtree("spec/phase1/output/05_c4_dedup")
        if os.path.exists("spec/phase1/logs/05_c4_dedup"):
            shutil.rmtree("spec/phase1/logs/05_c4_dedup")
        dedup_c4_data()
    else:
        # Clean previous outputs
        for folder in ["05_dedup_hash", "05_dedup_exact", "05_dedup_minhash", "05_dedup_bloom", "05_minhash_sigs"]:
            output_path = f"spec/phase1/output/{folder}"
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

        for folder in ["05_dedup_hash", "05_dedup_exact", "05_minhash_sig", "05_minhash_filter"]:
            log_path = f"spec/phase1/logs/{folder}"
            if os.path.exists(log_path):
                shutil.rmtree(log_path)

        # Create sample data
        total_docs = create_sample_data_with_duplicates()

        # Run different deduplication methods
        hash_based_dedup()

        # Note: Bloom filter and MinHash require more setup for local examples
        # Commenting out for now as they need proper initialization
        # exact_dedup_pipeline()
        # minhash_dedup_pipeline()

        # Analyze results
        analyze_results()