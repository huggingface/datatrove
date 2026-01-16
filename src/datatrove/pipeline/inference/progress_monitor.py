"""
Progress monitoring for dataset generation with periodic dataset card updates.

This module provides the ProgressMonitor PipelineStep that:
- Monitors the HuggingFace dataset repository for uploaded files
- Calculates progress based on documents processed
- Updates the dataset card with a progress bar and ETA
"""

import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pyarrow.parquet as pq
from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfFileSystem, list_repo_files
from huggingface_hub.errors import HfHubHTTPError

from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.dataset_card_generator import (
    InferenceDatasetCardParams,
    build_and_upload_dataset_card,
    format_number,
)
from datatrove.utils.logging import logger


def format_time_remaining(seconds: float) -> str:
    """
    Convert seconds to human-readable format.

    Examples:
        - 90 -> "1m"
        - 3600 -> "1h"
        - 5400 -> "1h 30m"
        - 7200 -> "2h"
    """
    if seconds < 60:
        return "< 1m"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)

    if hours > 0 and minutes > 0:
        return f"{hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h"
    else:
        return f"{minutes}m"


def format_completion_datetime(timestamp: float) -> str:
    """
    Format completion timestamp as readable date/time.

    Example: "Nov 27, 18:30 UTC"
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime("%b %d, %H:%M UTC")


def repo_has_parquet_data(repo_id: str) -> bool:
    """
    Check whether a dataset repo contains any parquet data files.

    Returns:
        True if at least one parquet file is found, False otherwise.
    """
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    except HfHubHTTPError as e:
        logger.warning(f"Could not list files for {repo_id}: {e}")
        return False

    has_parquet = any(file.lower().endswith(".parquet") for file in files)
    if not has_parquet:
        logger.info(f"No parquet files found yet in {repo_id}; skipping dataset load")
    return has_parquet


def count_documents_in_repo(repo_id: str) -> int:
    """
    Count total documents in uploaded parquet files in HF repo by reading parquet metadata.

    This approach reads only the file headers (a few KB per file) without downloading
    the actual data, making it efficient for large datasets and avoiding disk bloat.

    Returns 0 if repo doesn't exist or has no data yet.
    """
    try:
        # Create a fresh HfFileSystem instance to avoid caching issues
        # The filesystem caches directory listings, which prevents us from seeing new files
        fs = HfFileSystem()

        # Invalidate the cache for this specific path to get fresh file listing
        cache_path = f"datasets/{repo_id}/data"
        fs.invalidate_cache(cache_path)

        # Find all parquet files in the repo's data directory
        parquet_files = fs.glob(f"datasets/{repo_id}/data/*.parquet")

        if not parquet_files:
            logger.info(f"No parquet files found yet in {repo_id}")
            return 0

        total_rows = 0
        for file_path in parquet_files:
            try:
                # Invalidate cache for individual files too
                fs.invalidate_cache(file_path)
                with fs.open(file_path, "rb") as f:
                    parquet_file = pq.ParquetFile(f)
                    total_rows += parquet_file.metadata.num_rows
            except Exception as e:
                # File might be partially written; skip it
                logger.warning(f"Could not read metadata from {file_path}: {e}")
                continue

        logger.info(f"Counted {total_rows:,} documents in {repo_id} (from {len(parquet_files)} parquet files)")
        return total_rows
    except Exception as e:
        logger.warning(f"Could not count documents in {repo_id}: {e}")
        return 0


def get_total_expected_documents(dataset_name: str, split: str, config: str | None, max_examples: int) -> int:
    """
    Get total expected documents from input dataset.

    Args:
        dataset_name:   Name of the input dataset
        split:          Dataset split (e.g., 'train')
        config:         Optional dataset config
        max_examples:   Max examples to process (-1 for all)

    Returns:
        Total number of documents expected to be processed
    """
    # Try builder metadata first (fastest)
    try:
        builder = load_dataset_builder(dataset_name, name=config)
        if split in builder.info.splits:
            total = builder.info.splits[split].num_examples or 0
            if total > 0:
                logger.info(f"Found {total:,} documents from builder metadata")
                return min(total, max_examples) if max_examples > 0 else total
    except Exception as e:
        logger.warning(f"Could not get metadata from {dataset_name}: {e}")

    # Fall back to loading dataset directly
    try:
        logger.info(f"Loading dataset {dataset_name} to count rows...")
        ds = load_dataset(dataset_name, name=config, split=split)
        total = len(ds)
        logger.info(f"Dataset has {total:,} rows in {split} split")
        return min(total, max_examples) if max_examples > 0 else total
    except Exception as e:
        logger.warning(f"Could not load dataset {dataset_name}: {e}")

    # If we have a limit, use it as fallback
    if max_examples > 0:
        logger.info(f"Using max_examples {max_examples} as fallback estimate")
        return max_examples

    # If all else fails, raise an error
    raise ValueError(
        f"Could not determine dataset size for {dataset_name} (split={split}, config={config}). "
        f"Please ensure the dataset exists and is accessible, or specify --max-examples."
    )


def calculate_eta(completed: int, total: int, elapsed_time: float) -> tuple[float, datetime]:
    """
    Calculate estimated time remaining and completion datetime.

    Args:
        completed: Number of documents completed
        total: Total number of documents
        elapsed_time: Time elapsed since start (seconds)

    Returns:
        Tuple of (seconds_remaining, completion_datetime)
    """
    if completed == 0 or elapsed_time == 0:
        # Not enough data to estimate
        return 0.0, datetime.now(timezone.utc)

    docs_per_second = completed / elapsed_time
    remaining_docs = total - completed

    if docs_per_second == 0:
        return 0.0, datetime.now(timezone.utc)

    seconds_remaining = remaining_docs / docs_per_second
    completion_time = datetime.now(timezone.utc).timestamp() + seconds_remaining
    completion_dt = datetime.fromtimestamp(completion_time, tz=timezone.utc)

    return seconds_remaining, completion_dt


def render_progress_bar(completed: int, total: int, start_time: float, current_time: float) -> str:
    """
    Render progress bar with dots format.

    Format: [●●●●●●●●●●●●○○○○○○○○] 60% • 3,000/5,000 documents processed • ⏱️ 2h 15m remaining • 📅 Nov 27, 18:30 UTC
    """
    # Calculate percentage
    if total == 0:
        percentage = 0
    else:
        percentage = int((completed / total) * 100)

    # Create progress bar with 20 dots
    filled_dots = int((completed / total) * 20) if total > 0 else 0
    empty_dots = 20 - filled_dots
    bar = "[" + "●" * filled_dots + "○" * empty_dots + "]"

    # Format document count
    doc_text = f"{format_number(completed)}/{format_number(total)} documents processed"

    progress_text = f"{bar} {percentage}% • {doc_text}"

    # Calculate ETA
    elapsed_time = current_time - start_time
    if completed > 0 and completed < total:
        seconds_remaining, completion_dt = calculate_eta(completed, total, elapsed_time)
        time_text = f"⏱️ {format_time_remaining(seconds_remaining)} remaining"
        date_text = f"📅 {format_completion_datetime(completion_dt.timestamp())}"
        return f"{progress_text} • {time_text} • {date_text}"
    elif completed == 0:
        return f"{progress_text} • ⏱️ calculating..."
    else:
        return f"{progress_text} • ✅ Complete"


def create_progress_section_markdown(completed: int, total: int, start_time: float, current_time: float) -> str:
    """
    Create the full progress section for the dataset card.

    Returns markdown text to be inserted into the dataset card.
    """
    progress_bar = render_progress_bar(completed, total, start_time, current_time)

    return f"""
## 🔄 Generation Progress

{progress_bar}

*Last updated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}*
"""


@dataclass
class InferenceProgressMonitor(PipelineStep):
    """
    Monitor dataset generation progress and update dataset card periodically.

    This step:
    1. Monitors the HF dataset repo for uploaded files
    2. Calculates progress based on documents processed
    3. Updates the dataset card with progress bar and ETA
    4. Sleeps for update_interval seconds
    5. Repeats until stats.json is detected
    6. Generates final dataset card when complete
    """

    # Dataset card parameters
    params: InferenceDatasetCardParams

    # Monitoring parameters
    inference_job_id: str | None = None
    max_examples: int = -1
    update_interval: int = 3600  # 1 hour

    name: str = "InferenceProgressMonitor"
    type: str = "Monitor"

    def _is_job_running(self, job_id: str) -> bool:
        """Check if a Slurm job is still running or pending."""
        try:
            # -h removes header, -j specifies job
            result = subprocess.run(["squeue", "-h", "-j", job_id], capture_output=True, text=True)
            # If output is not empty, job is still in the queue (R, PD, etc.)
            return bool(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Warning: Failed to check Slurm job status: {e}")
            return True  # Assume running if check fails to avoid premature exit

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        """
        Monitor progress and update dataset card until completion.

        Only runs on rank 0. Yields data if provided (passthrough).
        """
        # Only run on rank 0
        if rank != 0:
            if data:
                yield from data
            return

        # Pass through data if provided
        if data:
            yield from data

        logger.info(f"Starting progress monitor for {self.params.output_repo_id}")
        if self.inference_job_id:
            logger.info(f"Monitoring inference job: {self.inference_job_id}")
        logger.info(f"Update interval: {self.update_interval} seconds")

        # Get total expected documents
        total_docs = get_total_expected_documents(
            self.params.input_dataset_name,
            self.params.input_dataset_split,
            self.params.input_dataset_config,
            self.max_examples,
        )
        logger.info(f"Total expected documents: {format_number(total_docs)}")

        start_time = time.time()

        while True:
            # Check if generation is complete (stats.json exists)
            if self.params.stats_path.exists():
                logger.info("stats.json detected - generation complete!")
                break

            # Check if inference job is still running (Slurm only)
            # Since the monitor runs in parallel with inference (no Slurm dependency),
            # it must manually check if the inference job has failed or stopped without
            # producing stats.json, otherwise it would run indefinitely.
            if self.inference_job_id and not self._is_job_running(self.inference_job_id):
                logger.info(
                    f"Inference job {self.inference_job_id} is no longer running and stats.json was not found. Stopping monitor."
                )
                break

            # Get current progress
            current_time = time.time()
            completed_docs = count_documents_in_repo(self.params.output_repo_id)

            logger.info(f"Progress: {format_number(completed_docs)}/{format_number(total_docs)} documents")

            # Create progress section
            try:
                progress_section = create_progress_section_markdown(
                    completed_docs, total_docs, start_time, current_time
                )

                # Update dataset card with progress (without waiting for stats.json)
                build_and_upload_dataset_card(
                    params=self.params,
                    progress_section=progress_section,
                )
                logger.info("Dataset card updated with progress")
            except Exception as e:
                logger.warning(f"Warning: Failed to update dataset card: {e}")

            # Sleep before next check
            logger.info(f"Sleeping for {self.update_interval} seconds...")
            time.sleep(self.update_interval)
