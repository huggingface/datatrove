"""
Example 4: Statistics Collection
=================================
Collect and analyze various statistics about documents in the dataset.

This example demonstrates:
- Document-level statistics (length, metadata)
- Word statistics (unique words, frequency distributions)
- Line statistics (lines per document)
- Language detection statistics
"""

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.stats import (
    DocStats,
    WordStats,
    LineStats,
    LangStats,
    TopKConfig,
)


def main():
    # Create the processing pipeline
    pipeline = [
        # Read from the tokenized output of Example 3
        JsonlReader(
            data_folder="spec/phase1/output/03_tokenized",
            glob_pattern="*.jsonl",
        ),

        # Collect various statistics
        DocStats(
            output_folder="spec/phase1/output/04_stats",
            histogram_round_digits=1,
        ),
        WordStats(
            output_folder="spec/phase1/output/04_stats",
            histogram_round_digits=0,
            top_k_config=TopKConfig(top_k_groups=["fqdn", "suffix"], top_k=100),
        ),
        LineStats(
            output_folder="spec/phase1/output/04_stats",
            histogram_round_digits=0,
        ),
        LangStats(
            output_folder="spec/phase1/output/04_stats",
            language="en",
        ),
    ]

    print("Starting Example 4: Statistics Collection")
    print("=" * 50)
    print("Analyzing documents from Example 3 (tokenized C4 data)")

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,  # Single task for simplicity
        logging_dir="spec/phase1/logs/04_statistics"
    )

    executor.run()

    print("\n" + "=" * 50)
    print("Statistics collection completed!")
    print("Stats saved in: spec/phase1/output/04_stats/")


def analyze_statistics():
    """Analyze the collected statistics"""
    import json
    from pathlib import Path

    stats_dir = Path("spec/phase1/output/04_stats")

    if not stats_dir.exists():
        print("No stats found. Run the pipeline first.")
        return

    print("\n" + "=" * 50)
    print("Statistics Analysis")
    print("=" * 50)

    # Document length statistics
    print("\n📊 Document Statistics:")
    doc_length_file = stats_dir / "summary/length/00000.json"
    if doc_length_file.exists():
        with open(doc_length_file, 'r') as f:
            stats = json.load(f)["summary"]
            print(f"  Total characters: {stats['total']:,}")
            print(f"  Documents: {stats['n']}")
            print(f"  Average length: {stats['mean']:.0f} chars")
            print(f"  Min: {stats['min']} chars")
            print(f"  Max: {stats['max']:,} chars")

    # Whitespace ratio
    ws_file = stats_dir / "summary/white_space_ratio/00000.json"
    if ws_file.exists():
        with open(ws_file, 'r') as f:
            stats = json.load(f)["summary"]
            print(f"  Whitespace ratio: {stats['mean']:.2%}")

    # Line statistics
    print("\n📏 Line Statistics:")
    lines_file = stats_dir / "summary/n_lines/00000.json"
    if lines_file.exists():
        with open(lines_file, 'r') as f:
            stats = json.load(f)["summary"]
            print(f"  Total lines: {stats['total']:,}")
            print(f"  Average per doc: {stats['mean']:.1f} lines")
            print(f"  Min: {stats['min']} lines")
            print(f"  Max: {stats['max']} lines")

    # Language statistics
    print("\n🌐 Language Statistics:")
    lang_file = stats_dir / "summary/fasttext_en/00000.json"
    if lang_file.exists():
        with open(lang_file, 'r') as f:
            stats = json.load(f)["summary"]
            print(f"  English confidence (0-1 scale):")
            print(f"    Average: {stats['mean']:.3f}")
            print(f"    Min: {stats['min']:.3f}")
            print(f"    Max: {stats['max']:.3f}")

    # Word statistics (if we had access to top-k words)
    print("\n📝 Word Statistics:")
    print("  (Top-k word tracking configured but not displayed in summary stats)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_statistics()
    else:
        # Clear previous completions to force re-run
        import shutil
        import os
        if os.path.exists("spec/phase1/logs/04_statistics/completions"):
            shutil.rmtree("spec/phase1/logs/04_statistics/completions")

        main()
        analyze_statistics()