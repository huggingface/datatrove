import argparse
import json
import os.path

from loguru import logger

from datatrove.io import BaseInputDataFolder, BaseOutputDataFile
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
    args, file_args = parser.parse_known_args()
    stats_folder = BaseInputDataFolder.from_path(args.path)
    # output file
    extra_args = dict(extra_arg.split("=") for extra_arg in file_args)
    path = extra_args.pop("path", os.path.abspath(args.output))
    output_file = BaseOutputDataFile.from_path(path, **extra_args)

    stats = []
    for file in stats_folder.list_files():
        with file.open() as f:
            stats.append(PipelineStats.from_json(json.load(f)))
    merged = sum(stats, start=PipelineStats())
    with output_file.open() as f:
        merged.save_to_disk(f)
    logger.info(f"Processing complete. Results saved to {output_file.path}.")
    logger.info(merged)


if __name__ == "__main__":
    main()
