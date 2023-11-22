import argparse
import os.path

from loguru import logger

from datatrove.io import BaseInputDataFolder


parser = argparse.ArgumentParser("Fetch the log files of failed tasks.")

parser.add_argument(
    "path", type=str, nargs="?", help="Path to the logging folder. Defaults to current directory.", default=os.getcwd()
)


def main():
    args = parser.parse_args()
    BaseInputDataFolder.from_path(args.path)
    logger.info("Fetching list of completed tasks...")


if __name__ == "__main__":
    main()
