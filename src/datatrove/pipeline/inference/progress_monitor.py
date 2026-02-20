"""
Progress monitoring for dataset generation with periodic dataset card updates.

This module provides the ProgressMonitor PipelineStep that:
- Monitors the HuggingFace dataset repository for uploaded files
- Calculates progress based on documents processed
- Updates the dataset card with a progress bar and ETA
"""

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfFileSystem, hf_hub_download, upload_file

from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.dataset_card_generator import (
    InferenceDatasetCardParams,
    build_and_upload_dataset_card,
    format_number,
    patch_readme_configs,
    patch_readme_prompt,
)
from datatrove.utils.logging import logger


_PROGRESS_SECTION_HEADER = "## 🔄 Generation Progress"
_TIMESTAMP_LINE_PATTERN = re.compile(r"^\*Last updated:.*$", re.MULTILINE)
_CONFIG_NAME_PATTERN = re.compile(r"^\s*-\s*config_name:\s*(?P<name>[^\s#]+)\s*$", re.MULTILINE)


def format_time_remaining(seconds: float) -> str:
    """Convert seconds to human-readable format with appropriate units.

    Picks the two largest non-zero units for readability:
        - 90s       -> "1m"
        - 5400s     -> "1h 30m"
        - 90000s    -> "1d 1h"
        - 700000s   -> "1w 1d"
        - 3000000s  -> "1mo 4d"
        - 40000000s -> "1y 3mo"
    """
    if seconds < 60:
        return "< 1m"

    # Define units from largest to smallest
    units = [
        ("y", 365.25 * 24 * 3600),
        ("mo", 30.44 * 24 * 3600),
        ("w", 7 * 24 * 3600),
        ("d", 24 * 3600),
        ("h", 3600),
        ("m", 60),
    ]

    parts: list[str] = []
    remaining = seconds
    for label, unit_seconds in units:
        if remaining >= unit_seconds:
            count = int(remaining // unit_seconds)
            remaining %= unit_seconds
            parts.append(f"{count}{label}")
            if len(parts) == 2:
                break

    return " ".join(parts)


def format_completion_datetime(timestamp: float) -> str:
    """
    Format completion timestamp as readable date/time.

    Example: "Nov 27 2026, 18:30 UTC"
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime("%b %d %Y, %H:%M UTC")


def _bounded_completed(completed: int, total: int) -> int:
    """Clamp completed count to a valid display range."""
    if total <= 0:
        return 0
    if completed < 0:
        return 0
    if completed > total:
        return total
    return completed


def _download_readme(repo_id: str) -> str | None:
    """Download the current README.md content from the HF repo.

    Returns None if the README doesn't exist yet.
    """
    try:
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="dataset")
        return Path(readme_path).read_text(encoding="utf-8")
    except Exception:
        return None


def _upload_readme(repo_id: str, content: str) -> None:
    """Upload README.md content to the HF repo."""
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    finally:
        os.remove(tmp_path)


def count_documents_in_repo(repo_id: str, config_name: str = "default") -> int:
    """
    Count total documents in uploaded parquet files in HF repo by reading parquet metadata.

    This approach reads only the file headers (a few KB per file) without downloading
    the actual data, making it efficient for large datasets and avoiding disk bloat.

    Args:
        repo_id: HuggingFace dataset repository ID.
        config_name: Dataset config name. "default" searches data/, named configs search {config_name}/.

    Returns 0 if repo doesn't exist or has no data yet.
    """
    try:
        # Create a fresh HfFileSystem instance to avoid caching issues
        # The filesystem caches directory listings, which prevents us from seeing new files
        fs = HfFileSystem()

        # Parquet files live under data/ for default config, or {config_name}/ for named configs
        subdir = "data" if config_name == "default" else config_name
        cache_path = f"datasets/{repo_id}/{subdir}"
        fs.invalidate_cache(cache_path)

        # Find all parquet files in the repo's data directory
        parquet_files = fs.glob(f"datasets/{repo_id}/{subdir}/**/*.parquet")

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


def render_progress_bar(
    completed: int,
    total: int,
    start_time: float,
    current_time: float,
) -> str:
    """Render a progress bar with ETA.

    Format: [●●●●●●●●●●●●○○○○○○○○] 60% • 3,000/5,000 docs<br>⏱️ 2h 15m remaining • 📅 Nov 27, 18:30 UTC
    """
    bounded_completed = _bounded_completed(completed, total)
    bar_text = _render_bar_and_counts(bounded_completed, total)
    elapsed_time = current_time - start_time
    if 0 < bounded_completed < total:
        seconds_remaining, completion_dt = calculate_eta(bounded_completed, total, elapsed_time)
        time_text = f"⏱️ {format_time_remaining(seconds_remaining)} remaining"
        date_text = f"📅 {format_completion_datetime(completion_dt.timestamp())}"
        return f"{bar_text}<br>{time_text} • {date_text}"
    if bounded_completed >= total > 0:
        return f"{bar_text}<br>✅ Complete"
    return f"{bar_text}<br>⏱️ waiting for first shard upload..."


def _render_bar_and_counts(completed: int, total: int) -> str:
    """Render just the progress bar and document counts (no ETA)."""
    bounded_completed = _bounded_completed(completed, total)
    percentage = int((bounded_completed / total) * 100) if total > 0 else 0
    filled_dots = int((bounded_completed / total) * 20) if total > 0 else 0
    empty_dots = 20 - filled_dots
    bar = "[" + "●" * filled_dots + "○" * empty_dots + "]"
    doc_text = f"{format_number(bounded_completed)}/{format_number(total)} docs"
    return f"{bar} {percentage}% • {doc_text}"


def _render_timestamp_line() -> str:
    return f"*Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*"


def _upsert_config_progress_line(readme_content: str, config_name: str, new_config_line: str) -> str | None:
    """Insert or replace the progress line for a single config."""
    config_pattern = re.compile(rf"^\*\*{re.escape(config_name)}\*\*:.*$", re.MULTILINE)
    if config_pattern.search(readme_content):
        return config_pattern.sub(new_config_line, readme_content)

    if _PROGRESS_SECTION_HEADER not in readme_content:
        logger.warning("No progress section found in README; this is unexpected during monitoring")
        return None

    match = _TIMESTAMP_LINE_PATTERN.search(readme_content)
    if match:
        return readme_content[: match.start()] + new_config_line + "\n\n" + readme_content[match.start() :]

    # Append at end of progress section if no timestamp line exists yet.
    idx = readme_content.index(_PROGRESS_SECTION_HEADER)
    next_section = re.search(r"\n## [^🔄]", readme_content[idx + 1 :])
    insert_pos = idx + 1 + next_section.start() if next_section else len(readme_content)
    return readme_content[:insert_pos] + new_config_line + "\n\n" + readme_content[insert_pos:]


def _replace_timestamp_line(readme_content: str) -> str:
    """Refresh the Last updated timestamp line."""
    return _TIMESTAMP_LINE_PATTERN.sub(_render_timestamp_line(), readme_content)


def patch_readme_progress(
    readme_content: str,
    config_name: str,
    completed: int,
    total_per_config: int,
    start_time: float,
    current_time: float,
) -> str:
    """Patch only the owned config's progress line and the timestamp
    in an existing README.

    If the progress section or the config line doesn't exist yet, they are
    created / appended.

    Args:
        readme_content: Current full README text.
        config_name: The config this monitor owns (e.g. "math").
        completed: Number of documents completed for this config.
        total_per_config: Total expected documents for this config.
        start_time: When this monitor started (for ETA).
        current_time: Current timestamp.

    Returns:
        Updated README text.
    """
    new_bar = render_progress_bar(completed, total_per_config, start_time, current_time)
    new_config_line = f"**{config_name}**: {new_bar}"

    updated = _upsert_config_progress_line(readme_content, config_name, new_config_line)
    if updated is None:
        return readme_content

    return _replace_timestamp_line(updated)


def create_progress_section_markdown(
    config_name: str,
    completed: int,
    total_per_config: int,
    start_time: float,
    current_time: float,
) -> str:
    """Create the initial progress section for a single config.

    Used only when the README has no progress section yet (first update).
    Subsequent updates use patch_readme_progress instead.
    """
    bar = render_progress_bar(completed, total_per_config, start_time, current_time)
    lines = [
        "",
        _PROGRESS_SECTION_HEADER,
        "",
        f"**{config_name}**: {bar}",
        "",
        _render_timestamp_line(),
    ]
    return "\n".join(lines)


def _extract_config_names(readme_content: str, current_config: str) -> list[str]:
    """Extract config names from README frontmatter and prioritize current config."""
    config_names = [current_config]
    seen = {current_config}
    for match in _CONFIG_NAME_PATTERN.finditer(readme_content):
        config_name = match.group("name").strip()
        if config_name in ("all",):
            continue
        if config_name in seen:
            continue
        seen.add(config_name)
        config_names.append(config_name)
    return config_names


def _append_progress_section(readme_content: str, progress_section: str) -> str:
    """Append a fresh progress section to an existing README."""
    stripped_readme = readme_content.rstrip()
    stripped_section = progress_section.strip()
    if not stripped_readme:
        return f"{stripped_section}\n"
    return f"{stripped_readme}\n\n{stripped_section}\n"


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
            if Path(self.params.stats_path).exists():
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

            current_time = time.time()

            # Count documents for THIS config only
            my_config = self.params.prompt_template_name
            my_count = count_documents_in_repo(self.params.output_repo_id, my_config)
            logger.info(f"Progress ({my_config}): {format_number(my_count)}/{format_number(total_docs)} documents")

            try:
                readme_content = _download_readme(self.params.output_repo_id)

                if readme_content is None:
                    # First update: build the full card with an initial progress section
                    progress_section = create_progress_section_markdown(
                        my_config, my_count, total_docs, start_time, current_time
                    )
                    build_and_upload_dataset_card(
                        params=self.params,
                        progress_section=progress_section,
                    )
                else:
                    updated_readme = readme_content
                    if _PROGRESS_SECTION_HEADER not in updated_readme:
                        config_names = _extract_config_names(updated_readme, my_config)
                        progress_section = create_progress_section_markdown(
                            my_config, my_count, total_docs, start_time, current_time
                        )
                        for config_name in config_names[1:]:
                            config_count = count_documents_in_repo(self.params.output_repo_id, config_name)
                            progress_section = patch_readme_progress(
                                progress_section,
                                config_name,
                                config_count,
                                total_docs,
                                start_time,
                                current_time,
                            )
                        updated_readme = _append_progress_section(updated_readme, progress_section)

                    # Patch progress, prompt, and configs for this config
                    updated = patch_readme_progress(
                        updated_readme, my_config, my_count, total_docs, start_time, current_time
                    )
                    if self.params.prompt_template:
                        updated = patch_readme_prompt(updated, my_config, self.params.prompt_template)
                    updated = patch_readme_configs(
                        updated,
                        self.params.output_repo_id,
                        my_config,
                        self.params.input_dataset_split or "train",
                    )
                    _upload_readme(self.params.output_repo_id, updated)
                logger.info("Dataset card updated with progress")
            except Exception as e:
                logger.warning(f"Warning: Failed to update dataset card: {e}")

            # Sleep before next check
            logger.info(f"Sleeping for {self.update_interval} seconds...")
            time.sleep(self.update_interval)
