import argparse
import json
import os.path
import re
import subprocess

from rich.console import Console
from rich.prompt import Confirm

from datatrove.io import get_datafolder
from datatrove.utils._import_utils import is_rich_available
from datatrove.utils.logging import logger


if not is_rich_available():
    raise ImportError("Please install `rich` to run this command (`pip install rich`).")


parser = argparse.ArgumentParser("Fetch the log files of failed tasks.")

parser.add_argument(
    "path", type=str, nargs="?", help="Path to the logging folder. Defaults to current directory.", default=os.getcwd()
)

parser.add_argument(
    "--running-slurm-jobs", 
    type=str, 
    help="Comma-separated list of running slurm job IDs to exclude from failed logs", 
    default=""
)

RANK_FROM_LOG_FILENAME_REGEX = re.compile(r"logs/task_(\d{5})\.log")
RANK_FROM_PIPELINE_REGEX = re.compile(r"Launching pipeline for rank=(\d+)")


def get_running_ranks_from_slurm_jobs(job_ids, console):
    """
    Get ranks that are currently running from slurm job logs using scontrol show job.
    
    Args:
        job_ids: List of slurm job IDs
        console: Rich console for logging
    
    Returns:
        Set of ranks that are currently running
    """
    running_ranks = set()
    
    for job_id in job_ids:
        if not job_id.strip():
            continue
            
        try:
            # Use scontrol to get job information
            result = subprocess.run(['scontrol', 'show', 'job', job_id.strip()], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                console.log(f"[yellow]Warning: Could not get info for slurm job {job_id} (job may not exist or be finished)[/yellow]")
                continue
                
            # Parse scontrol output to find log files
            log_files = []
            for line in result.stdout.split('\n'):
                # Look for StdOut and StdErr paths
                if 'StdOut=' in line:
                    stdout_path = line.split('StdOut=')[1].split()[0]
                    if stdout_path != '/dev/null':
                        log_files.append(stdout_path)
                elif 'StdErr=' in line:
                    stderr_path = line.split('StdErr=')[1].split()[0] 
                    if stderr_path != '/dev/null':
                        log_files.append(stderr_path)
            
            # Read log files and search for running ranks
            job_running_ranks = set()
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                            # Find all ranks mentioned in "Launching pipeline for rank=X"
                            ranks = RANK_FROM_PIPELINE_REGEX.findall(log_content)
                            file_ranks = {int(rank) for rank in ranks}
                            job_running_ranks.update(file_ranks)
                    else:
                        console.log(f"[yellow]Warning: Log file {log_file} for job {job_id} does not exist[/yellow]")
                except Exception as e:
                    console.log(f"[yellow]Warning: Could not read log file {log_file} for job {job_id}: {e}[/yellow]")
            
            running_ranks.update(job_running_ranks)
            if job_running_ranks:
                console.log(f"Job {job_id}: found {len(job_running_ranks)} running ranks")
            else:
                console.log(f"[yellow]Job {job_id}: no running ranks found in logs[/yellow]")
                
        except subprocess.TimeoutExpired:
            console.log(f"[red]Error: Timeout when querying slurm job {job_id}[/red]")
        except Exception as e:
            console.log(f"[red]Error: Could not query slurm job {job_id}: {e}[/red]")
    
    return running_ranks


def main():
    """
    Takes a `logging_dir` as input, gets total number of tasks from `executor.json` and then gets which ranks are
    incomplete by scanning `logging_dir/completions`. The log files for the incomplete tasks are then displayed.
    
    Now also excludes ranks that are currently running in specified slurm jobs.
    """
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

    # Get running ranks from slurm jobs
    running_ranks = set()
    if args.running_slurm_jobs:
        job_ids = [job_id.strip() for job_id in args.running_slurm_jobs.split(',')]
        console.log(f"Checking running slurm jobs: {job_ids}")
        with console.status("Checking running slurm jobs for active ranks"):
            running_ranks = get_running_ranks_from_slurm_jobs(job_ids, console)
        console.log(f"Found {len(running_ranks)} ranks currently running in slurm jobs")

    with console.status("Fetching list of incomplete tasks"):
        completed = set(logging_dir.list_files("completions"))
        incomplete = set(filter(lambda rank: f"completions/{rank:05d}" not in completed, range(world_size)))
        
        # Remove running ranks from incomplete set
        incomplete = incomplete - running_ranks
        
    console.log(f"Found {len(incomplete)}/{world_size} incomplete tasks (excluding {len(running_ranks)} running tasks).")

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
