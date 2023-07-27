import argparse
import json
import os.path

from loguru import logger

from datatrove.utils.stats import PipelineStats


parser = argparse.ArgumentParser("Combine and average per task statistics into a single file.")

parser.add_argument(
    "path", type=str, nargs="?", help="Path to the stats folder. Defaults to current directory.", default=os.getcwd()
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    help="Save file location. Defaults to 'merged_stats.json'.",
    default="merged_stats.json",
)


def main():
    args = parser.parse_args()
    stats_path = args.path
    output_file = args.output

    stats = []
    for file in os.listdir(stats_path):
        with open(os.path.join(stats_path, file)) as f:
            stats.append(PipelineStats.from_json(json.load(f)))
    merged = sum(stats)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.save_to_disk(output_file)
    logger.info(f"Processing complete. Results saved to {output_file}.")


if __name__ == "__main__":
    main()
