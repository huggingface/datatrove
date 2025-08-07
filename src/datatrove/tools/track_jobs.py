import argparse
import glob
import json
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from datatrove.io import get_datafolder
from datatrove.utils._import_utils import is_rich_available
from datatrove.utils.logging import logger


if not is_rich_available():
    raise ImportError("Please install `rich` to run this command (`pip install rich`).")


parser = argparse.ArgumentParser("Track job progress with optional continuous monitoring.")

parser.add_argument("path", type=str, help="Path to the logging folder(s). May contain glob patterns like '*'.")

parser.add_argument("-i", "--interval", type=int, help="Refresh interval in seconds for continuous monitoring.")


def expand_path_pattern(path_pattern):
    """Expand glob pattern or return original path if no magic characters."""
    if any(char in path_pattern for char in ["*", "?", "["]):
        return glob.glob(path_pattern)
    else:
        return [path_pattern]


def find_valid_directories(paths, invalid_dirs_cache):
    """Find directories that contain executor.json."""
    valid_dirs = []
    for path in paths:
        # Skip paths we've already determined are invalid
        if path in invalid_dirs_cache:
            continue

        try:
            datafolder = get_datafolder(path)
            if datafolder.isfile("executor.json"):
                valid_dirs.append(path)
            else:
                # Cache as invalid if no executor.json
                invalid_dirs_cache.add(path)
        except Exception:
            # Cache as invalid if we can't access it
            invalid_dirs_cache.add(path)
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
        status = {"completed_tasks": completed_count, "total_tasks": world_size, "is_complete": is_complete}

        # Cache completed jobs to avoid re-checking
        if is_complete:
            completed_jobs_cache[job_path] = status

        return status

    except Exception:
        return None


def create_display(job_statuses, console, previous_state=None):
    """Create the display panel with progress information."""
    # Calculate job progress
    total_jobs = len(job_statuses)
    completed_jobs = sum(1 for status in job_statuses.values() if status and status["is_complete"])

    # Calculate task progress
    total_tasks = sum(status["total_tasks"] for status in job_statuses.values() if status)
    completed_tasks = sum(status["completed_tasks"] for status in job_statuses.values() if status)

    # Calculate deltas from previous state
    job_delta = ""
    task_delta = ""
    if previous_state:
        job_diff = completed_jobs - previous_state.get("completed_jobs", 0)
        task_diff = completed_tasks - previous_state.get("completed_tasks", 0)
        if job_diff > 0:
            job_delta = f" [bright_green](+{job_diff} completed)[/bright_green]"
        if task_diff > 0:
            task_delta = f" [bright_green](+{task_diff} completed)[/bright_green]"

    # Choose emoji based on completion
    job_emoji = "✅" if completed_jobs == total_jobs else "🔄"
    task_emoji = "✅" if completed_tasks == total_tasks else "🔄"

    # Calculate percentages
    job_percentage = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
    task_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

    # Create progress bar visualization using available terminal width
    # Account for padding, borders, and the text that follows the progress bar
    # Format: " completed/total (percentage%)" - roughly 20 characters max
    available_width = max(console.size.width - 35, 20)  # More conservative margin
    bar_width = available_width

    def create_progress_bar(
        current_percentage, previous_percentage=0, filled_char="█", new_char="▓", empty_char="░", cursor_char="▶"
    ):
        current_filled = int(current_percentage / 100 * bar_width)
        previous_filled = int(previous_percentage / 100 * bar_width)

        # Calculate different sections
        old_progress = min(previous_filled, current_filled)
        new_progress = max(0, current_filled - previous_filled)

        # Build the progress bar
        bar = ""

        # Add old progress
        bar += filled_char * old_progress

        # Add new progress (if any)
        if new_progress > 0:
            if current_filled < bar_width and current_percentage < 100:
                # Add new progress with cursor at the end
                bar += new_char * (new_progress - 1) + cursor_char
            else:
                bar += new_char * new_progress
        elif current_filled < bar_width and current_percentage < 100 and old_progress > 0:
            # No new progress, but add cursor if not complete
            bar = bar[:-1] + cursor_char

        # Fill remaining with empty characters
        bar += empty_char * (bar_width - len(bar))

        return bar

    # Calculate previous percentages for progress bar visualization
    prev_job_percentage = 0
    prev_task_percentage = 0
    if previous_state:
        prev_job_percentage = (
            (previous_state.get("completed_jobs", 0) / previous_state.get("total_jobs", 1) * 100)
            if previous_state.get("total_jobs", 0) > 0
            else 0
        )
        prev_task_percentage = (
            (previous_state.get("completed_tasks", 0) / previous_state.get("total_tasks", 1) * 100)
            if previous_state.get("total_tasks", 0) > 0
            else 0
        )

    job_bar = create_progress_bar(job_percentage, prev_job_percentage)
    task_bar = create_progress_bar(task_percentage, prev_task_percentage)

    # Get current timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Create timestamp line at top right
    timestamp_line = f"[dim]Last update: {timestamp}[/dim]"
    # Calculate spacing to right-align the timestamp more accurately
    # Account for panel borders (4 chars) and padding (2 chars each side)
    panel_overhead = 8
    available_width = max(console.size.width - panel_overhead, 20)
    timestamp_text_len = len(f"Last update: {timestamp}")  # Without markup
    spaces_needed = max(0, available_width - timestamp_text_len)
    right_aligned_timestamp = f"{' ' * spaces_needed}{timestamp_line}"

    # Create the main content
    main_content = f"""
{job_emoji} [bold]Tracked jobs: {completed_jobs}/{total_jobs} - {job_percentage:.1f}%{job_delta}[/bold]
[green]{job_bar}[/green]

{task_emoji} [bold]Tracked tasks: {completed_tasks}/{total_tasks} - {task_percentage:.1f}%{task_delta}[/bold]
[cyan]{task_bar}[/cyan]
    """.strip()

    content_with_timestamp = f"{right_aligned_timestamp}\n\n{main_content}"

    # Return both the panel and current state for next comparison
    current_state = {
        "completed_jobs": completed_jobs,
        "completed_tasks": completed_tasks,
        "total_jobs": total_jobs,
        "total_tasks": total_tasks,
    }

    return Panel(content_with_timestamp, title="Job Tracking Status", border_style="blue"), current_state


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
    invalid_dirs_cache = set()
    valid_dirs = find_valid_directories(expanded_paths, invalid_dirs_cache)

    if not valid_dirs:
        console.print("[red]Error: No valid job directories found (directories must contain executor.json)[/red]")
        return

    completed_jobs_cache = {}
    previous_state = None

    def update_display():
        """Update and return the current display."""
        nonlocal valid_dirs, previous_state
        # Re-expand glob patterns to catch new matches
        current_expanded = expand_path_pattern(args.path)
        # Only check new paths that aren't already known to be valid or invalid
        new_paths = [p for p in current_expanded if p not in valid_dirs and p not in invalid_dirs_cache]
        if new_paths:
            new_valid = find_valid_directories(new_paths, invalid_dirs_cache)
            valid_dirs.extend(new_valid)

        job_statuses = {}
        for job_path in valid_dirs:
            job_statuses[job_path] = get_job_status(job_path, completed_jobs_cache)

        panel, current_state = create_display(job_statuses, console, previous_state)
        previous_state = current_state
        return panel

    if args.interval:
        # Continuous monitoring mode
        console.clear()  # Clear terminal at start
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
