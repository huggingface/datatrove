import argparse
import glob
import json
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from datatrove.io import get_datafolder
from datatrove.utils._import_utils import is_rich_available
from datatrove.utils.logging import logger


if not is_rich_available():
    raise ImportError("Please install `rich` to run this command (`pip install rich`).")


parser = argparse.ArgumentParser("Track job progress with optional continuous monitoring.")

parser.add_argument(
    "path", type=str, help="Path to the logging folder(s). May contain glob patterns like '*'."
)

parser.add_argument(
    "-i", "--interval", type=int, help="Refresh interval in seconds for continuous monitoring."
)


def expand_path_pattern(path_pattern):
    """Expand glob pattern or return original path if no magic characters."""
    if any(char in path_pattern for char in ['*', '?', '[']):
        return glob.glob(path_pattern)
    else:
        return [path_pattern]


def find_valid_directories(paths):
    """Find directories that contain executor.json."""
    valid_dirs = []
    for path in paths:
        try:
            datafolder = get_datafolder(path)
            if datafolder.isfile("executor.json"):
                valid_dirs.append(path)
        except Exception:
            # Skip invalid paths
            continue
    return valid_dirs


def get_job_status(job_path, completed_jobs_cache):
    """Get completion status for a job directory."""
    # If already marked as complete, skip re-checking
    if job_path in completed_jobs_cache:
        return completed_jobs_cache[job_path]
    
    try:
        logging_dir = get_datafolder(job_path)
        
        with logging_dir.open("executor.json", "rt") as f:
            world_size = json.load(f).get("world_size", None)
            
        if not world_size:
            return None
            
        completed = set(logging_dir.list_files("completions"))
        completed_count = len(completed)
        
        is_complete = completed_count == world_size
        status = {
            'completed_tasks': completed_count,
            'total_tasks': world_size,
            'is_complete': is_complete
        }
        
        # Cache completed jobs to avoid re-checking
        if is_complete:
            completed_jobs_cache[job_path] = status
            
        return status
        
    except Exception:
        return None


def create_display(job_statuses):
    """Create the display panel with progress information."""
    # Calculate job progress
    total_jobs = len(job_statuses)
    completed_jobs = sum(1 for status in job_statuses.values() if status and status['is_complete'])
    
    # Calculate task progress
    total_tasks = sum(status['total_tasks'] for status in job_statuses.values() if status)
    completed_tasks = sum(status['completed_tasks'] for status in job_statuses.values() if status)
    
    # Choose emoji based on completion
    job_emoji = "✅" if completed_jobs == total_jobs else "❌"
    task_emoji = "✅" if completed_tasks == total_tasks else "❌"
    
    # Calculate percentages
    job_percentage = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
    task_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    # Create progress bar visualization
    bar_width = 30
    job_filled = int(job_percentage / 100 * bar_width)
    task_filled = int(task_percentage / 100 * bar_width)
    
    job_bar = "█" * job_filled + "░" * (bar_width - job_filled)
    task_bar = "█" * task_filled + "░" * (bar_width - task_filled)
    
    # Create the display content
    content = f"""
{job_emoji} [bold]Tracked jobs: {total_jobs}[/bold]
[blue]{job_bar}[/blue] {completed_jobs}/{total_jobs} ({job_percentage:.0f}%)

{task_emoji} [bold]Tracked tasks: {total_tasks}[/bold]
[blue]{task_bar}[/blue] {completed_tasks}/{total_tasks} ({task_percentage:.0f}%)
    """.strip()
    
    return Panel(content, title="Job Tracking Status", border_style="blue")


def main():
    """
    Track job progress with optional continuous monitoring.
    """
    args = parser.parse_args()
    console = Console()
    logger.remove()
    
    # Expand glob pattern
    expanded_paths = expand_path_pattern(args.path)
    
    # Find valid directories
    valid_dirs = find_valid_directories(expanded_paths)
    
    if not valid_dirs:
        console.print("[red]Error: No valid job directories found (directories must contain executor.json)[/red]")
        return
    
    completed_jobs_cache = {}
    
    def update_display():
        """Update and return the current display."""
        job_statuses = {}
        for job_path in valid_dirs:
            job_statuses[job_path] = get_job_status(job_path, completed_jobs_cache)
        return create_display(job_statuses)
    
    if args.interval:
        # Continuous monitoring mode
        try:
            with Live(update_display(), refresh_per_second=1) as live:
                while True:
                    time.sleep(args.interval)
                    live.update(update_display())
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped.[/yellow]")
    else:
        # Single update mode
        console.print(update_display())


if __name__ == "__main__":
    main()
