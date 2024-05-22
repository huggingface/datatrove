import argparse
import json
import os.path

from rich.console import Console

from datatrove.io import get_datafolder
from datatrove.utils._import_utils import is_rich_available
from datatrove.utils.logging import logger


if not is_rich_available():
    raise ImportError("Please install `rich` to run this command (`pip install rich`).")


parser = argparse.ArgumentParser("Fetch all jobs that are running or complete.")

parser.add_argument(
    "path", type=str, nargs="?", help="Path to the logging folder. Defaults to current directory.", default=os.getcwd()
)

parser.add_argument(
    "-p", "--log_prefix", type=str, nargs="?", help="Prefix of logging folders to be scanned.", default=""
)
parser.add_argument("-hc", "--hide_complete", help="Hide all jobs that are already complete.", action="store_true")


def main():
    """
    Takes a `path` as input, gets all valid job folders and their total number of tasks from `executor.json` and then gets which ranks are
    incomplete by scanning `path/{LOGGING_DIRS}/completions`. If a `log_prefix` is provided the directories following the `path/log_prefix{LOGGING_DIRS}/completions`
    pattern are scanned.
    """
    args = parser.parse_args()
    console = Console()

    main_folder = get_datafolder(args.path)
    logging_dirs = [
        f
        for f, info in main_folder.glob(f"{args.log_prefix}*", detail=True, maxdepth=1).items()
        if info["type"] == "directory"
    ]
    logger.remove()

    complete_jobs = 0
    incomplete_jobs = 0

    for path in logging_dirs:
        logging_dir = get_datafolder(main_folder.resolve_paths(path))
        if not logging_dir.isfile("executor.json"):
            console.log(
                f'Could not find "executor.json" in the given directory ({path}). Are you sure it is a '
                "logging folder?",
                style="red",
            )
            continue
        with logging_dir.open("executor.json", "rt") as f:
            world_size = json.load(f).get("world_size", None)
        if not world_size:
            console.log(
                f"Could not get the total number of tasks in {path}, please try relaunching the run.",
                style="red",
            )
            continue

        with console.status("Fetching list of incomplete tasks"):
            completed = set(logging_dir.list_files("completions"))
            incomplete = set(filter(lambda rank: f"completions/{rank:05d}" not in completed, range(world_size)))

        if len(incomplete) == 0:
            emoji = "✅"
            complete_jobs += 1
        else:
            emoji = "❌"
            incomplete_jobs += 1

        if len(incomplete) > 0 or not args.hide_complete:
            console.log(
                f"{emoji} {path + ':': <50}{len(completed)}/{world_size} ({len(completed)/(world_size):.0%}) completed tasks."
            )

    if complete_jobs + incomplete_jobs > 0:
        console.log(
            f"Summary: {complete_jobs}/{complete_jobs+incomplete_jobs} ({complete_jobs/(complete_jobs+incomplete_jobs):.0%}) jobs completed."
        )
    else:
        console.log("No jobs found.")


if __name__ == "__main__":
    main()
