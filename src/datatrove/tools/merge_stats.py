import argparse
import json
import os.path

from tqdm import tqdm

from datatrove.io import get_datafolder, open_file
from datatrove.utils.logging import logger
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
    stats_folder = get_datafolder(args.path)
    # output file
    path = args.output

    stats = []
    for file in tqdm(stats_folder.list_files()):
        with stats_folder.open(file, "rt") as f:
            stats.append(PipelineStats.from_json(json.load(f)))
    merged = sum(tqdm(stats), start=PipelineStats())
    with open_file(path, mode="wt") as f:
        merged.save_to_disk(f)
    logger.info(f"Processing complete. Results saved to {path}.")
    logger.info(merged)


if __name__ == "__main__":
    main()
