import argparse
import json
import os.path
import re

from loguru import logger
from rich.console import Console
from rich.prompt import Confirm

from datatrove.io import get_datafolder
from datatrove.utils._import_utils import is_rich_available

def list_folders_with_prefix(log_files_path, log_prefix):
    # Get a list of all folders in the given path
    folders = [folder for folder in os.listdir(log_files_path) if os.path.isdir(os.path.join(log_files_path, folder))]
    # Filter out only the folders that start with the specified prefix
    folders_with_prefix = [os.path.join(log_files_path, folder) for folder in folders if folder.startswith(log_prefix)]
    
    return folders_with_prefix

if not is_rich_available():
    raise ImportError("Please install `rich` to run this command (`pip install rich`).")


parser = argparse.ArgumentParser("Fetch all jobs that are running or complete.")

parser.add_argument(
    "path", type=str, nargs="?", help="Path to the logging folder. Defaults to current directory.", default=os.getcwd()
)

parser.add_argument(
    "--log_prefix", type=str, nargs="?", help="Prefix of logging folders to be scanned.", default=""
)
parser.add_argument('--show_complete', help="Also list all jobs that are already complete.", action='store_true')
RANK_FROM_LOG_FILENAME_REGEX = re.compile(r"logs/task_(\d{5})\.log")


def main():
    """
    Takes a `path` as input, gets all valid job folders and their total number of tasks from `executor.json` and then gets which ranks are
    incomplete by scanning `path/{LOGGING_DIRS}/completions`. If a `log_prefix` is provided the directories following the `path/log_prefix{LOGGING_DIRS}/completions`
    pattern are scanned.
    """
    args = parser.parse_args()
    console = Console()

    logging_dirs = sorted(list_folders_with_prefix(args.path, args.log_prefix))
    logger.remove()

    complete_jobs = 0
    incomplete_jobs = 0

    for path in logging_dirs:
        logging_dir = get_datafolder(path)
        if not logging_dir.isfile("executor.json"):
            console.log(
                'Could not find "executor.json" in the given directory. Are you sure it is a ' "logging folder?",
                style="red",
            )
            continue
        with logging_dir.open("executor.json", "rt") as f:
            world_size = json.load(f).get("world_size", None)
        if not world_size:
            console.log("Could not get the total number of tasks, please try relaunching the run.", style="red")
            continue 

        with console.status("Fetching list of incomplete tasks"):
            completed = set(logging_dir.list_files("completions"))
            incomplete = set(filter(lambda rank: f"completions/{rank:05d}" not in completed, range(world_size)))

        if len(incomplete)==0:
            emoji = "✅"
            complete_jobs += 1
        else:
            emoji = "❌"
            incomplete_jobs += 1
        
        if not (len(incomplete)==0 and not args.show_complete):
            console.log(f"{emoji}{path.split('/')[-1]+':': <50}{len(completed)}/{world_size} completed tasks.")

    console.log(f"Summary: {complete_jobs}/{complete_jobs+incomplete_jobs} jobs completed.")

if __name__ == "__main__":
    main()