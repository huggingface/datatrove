import argparse
import json
import os.path
import re

from loguru import logger
from rich.console import Console
from rich.prompt import Confirm

from datatrove.io import get_datafolder


parser = argparse.ArgumentParser("Fetch the log files of failed tasks.")

parser.add_argument(
    "path", type=str, nargs="?", help="Path to the logging folder. Defaults to current directory.", default=os.getcwd()
)

RANK_FROM_LOG_FILENAME_REGEX = re.compile(r"logs/task_(\d{5})\.log")


def main():
    args = parser.parse_args()
    console = Console()

    logger.remove()

    logging_dir = get_datafolder(args.path)
    if not logging_dir.isfile("executor.json"):
        console.log(
            'Could not find "executor.json" in the given directory. Are you sure it is a ' "logging folder?",
            style="red",
        )
        return
    with logging_dir.open("executor.json", "rt") as f:
        world_size = json.load(f).get("world_size", None)
    if not world_size:
        console.log("Could not get the total number of tasks, please try relaunching the run.", style="red")
        return
    console.log(f"Found executor config: {world_size} tasks")

    with console.status("Fetching list of incomplete tasks"):
        completed = set(logging_dir.list_files("completions"))
        incomplete = set(filter(lambda rank: f"completions/{rank:05d}" not in completed, range(world_size)))
    console.log(f"Found {len(incomplete)}/{world_size} incomplete tasks.")

    with console.status("Looking for log files"):
        incomplete_logs = list(
            filter(
                lambda file: int(RANK_FROM_LOG_FILENAME_REGEX.search(file).group(1)) in incomplete,
                logging_dir.list_files("logs"),
            )
        )
    console.log(f"Found {len(incomplete_logs)} log files for incomplete tasks.")
    first = True
    for incomplete_log in incomplete_logs:
        if not first and not Confirm.ask(f"Show next log ([i]{incomplete_log}[/i])?", default=True):
            break
        with console.pager():
            with logging_dir.open(incomplete_log, "rt") as f:
                console.print(f.read())
        first = False


if __name__ == "__main__":
    main()
