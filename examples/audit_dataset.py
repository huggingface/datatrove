"""Run a small, local dataset audit with reviewable filtering artifacts.

Usage:
    python examples/audit_dataset.py --work-dir /tmp/datatrove-audit

The length threshold is only an example. Choose quality rules from the intended
dataset and model use case, then inspect both kept and rejected documents.
"""

import argparse
import json
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.stats import DocStats, StatsMerger
from datatrove.pipeline.writers import JsonlWriter


MIN_CHARS = 80


def write_sample_input(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    documents = [
        {"id": "short", "text": "Limited offer. Buy now!", "source": "demo", "dump": "2026-08"},
        {
            "id": "guide",
            "text": "A dataset audit keeps filtering decisions reviewable by saving accepted and rejected records.",
            "source": "demo",
            "dump": "2026-08",
        },
        {
            "id": "provenance",
            "text": "Provenance fields record where a document came from and which source snapshot was processed.",
            "source": "demo",
            "dump": "2026-08",
        },
    ]
    with (input_dir / "sample.jsonl").open("w") as output:
        for document in documents:
            output.write(json.dumps(document) + "\n")


def merge_stats(stats_dir: Path) -> None:
    list(StatsMerger(input_folder=stats_dir, output_folder=stats_dir)())


def main(work_dir: Path) -> None:
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    stats_dir = work_dir / "stats"
    write_sample_input(input_dir)

    executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(input_dir, glob_pattern="*.jsonl"),
            DocStats(stats_dir / "before", groups_to_compute=["summary", "histogram"]),
            LambdaFilter(
                lambda document: len(document.text) >= MIN_CHARS,
                exclusion_writer=JsonlWriter(output_dir / "removed", compression=None),
            ),
            DocStats(stats_dir / "after", groups_to_compute=["summary", "histogram"]),
            JsonlWriter(output_dir / "kept", compression=None),
        ],
        tasks=1,
        workers=1,
        logging_dir=work_dir / "logs",
    )
    executor.run()

    merge_stats(stats_dir / "before")
    merge_stats(stats_dir / "after")
    print(f"Audit artifacts written to {work_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("dataset-audit"))
    main(parser.parse_args().work_dir)
