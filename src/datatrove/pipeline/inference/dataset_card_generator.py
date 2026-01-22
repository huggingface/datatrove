import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import dataset_info, upload_file, whoami

from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger


@dataclass
class InferenceDatasetCardParams:
    """Parameters required for generating a dataset card."""

    output_repo_id: str
    input_dataset_name: str
    input_dataset_split: str
    input_dataset_config: str | None
    prompt_column: str
    prompt_template: str | None
    system_prompt: str | None
    model_name: str
    model_revision: str
    generation_kwargs: dict[str, Any]
    spec_config: str | None
    stats_path: Path


@dataclass
class InferenceDatasetCardGenerator(PipelineStep):
    """Generate final dataset card after inference completes."""

    params: InferenceDatasetCardParams

    name: str = "InferenceDatasetCardGenerator"
    type: str = "Generator"

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        """
        Generate final dataset card after all data is processed.

        Only runs on rank 0. Yields data if provided (passthrough).
        """
        if data:
            yield from data

        if rank != 0:
            return

        logger.info(f"Generating final dataset card for {self.params.output_repo_id}...")
        try:
            build_and_upload_dataset_card(
                params=self.params,
                progress_section="",  # No progress section in final card
            )
            logger.info("Final dataset card generated successfully")
        except Exception as e:
            logger.warning(f"Warning: Failed to generate final dataset card: {e}")


@dataclass
class JobStats:
    document_count: int
    mean_doc_len: float | None
    prompt_tokens_total: int | None
    completion_tokens_total: int | None
    prompt_tokens_mean: float | None
    completion_tokens_mean: float | None


def load_job_stats(stats_path: Path, timeout: int = 60 * 5) -> JobStats | None:
    start_time = time.time()
    logger.info(f"Waiting for stats file at {stats_path} (timeout: {timeout}s)...")
    while not stats_path.exists():
        if time.time() - start_time > timeout:
            logger.warning(f"Timeout waiting for {stats_path}")
            return None
        logger.info(f"Stats file not found yet at {stats_path}, waiting... ({int(time.time() - start_time)}s elapsed)")
        time.sleep(10)
    logger.info(f"Stats file found at {stats_path}")

    time.sleep(2)  # Give a small buffer for write completion

    try:
        with stats_path.open() as f:
            entries = json.load(f)
        logger.info(f"Successfully loaded stats from {stats_path}")
    except Exception as e:
        logger.error(f"Failed to load stats file: {e}")
        return None

    doc_entry = next((entry for entry in entries if "doc_len" in entry.get("stats", {})), None)
    model_entry = next((entry for entry in entries if "prompt_tokens" in entry.get("stats", {})), None)

    if not doc_entry:
        return None

    doc_stats = doc_entry["stats"]["doc_len"]
    prompt_stats = model_entry["stats"]["prompt_tokens"] if model_entry else {}
    completion_stats = model_entry["stats"]["completion_tokens"] if model_entry else {}

    # Handle doc_len - can be dict with stats or just a number
    if isinstance(doc_stats, dict):
        doc_len_n = doc_stats.get("n", 0)
        mean_doc_len = doc_stats.get("mean")
    else:
        doc_len_n = doc_stats if isinstance(doc_stats, int) else 0
        mean_doc_len = None

    # Handle documents count - can be dict with 'total' or just a number
    documents = doc_entry["stats"].get("documents")
    if isinstance(documents, dict):
        document_count = documents.get("total", doc_len_n)
    else:
        document_count = documents or doc_len_n

    return JobStats(
        document_count=document_count,
        mean_doc_len=mean_doc_len,
        prompt_tokens_total=prompt_stats.get("total"),
        completion_tokens_total=completion_stats.get("total"),
        prompt_tokens_mean=prompt_stats.get("mean"),
        completion_tokens_mean=completion_stats.get("mean"),
    )


def fetch_source_dataset_metadata(dataset_name: str) -> dict[str, Any]:
    logger.info(f"Fetching metadata for source dataset: {dataset_name}")

    info = dataset_info(dataset_name, expand=["cardData"])
    card_data = getattr(info, "card_data", None) or getattr(info, "cardData", None) or {}

    return {
        "license": getattr(info, "license", None) or card_data.get("license"),
        "languages": card_data.get("language") or card_data.get("languages") or [],
        "tags": card_data.get("tags") or [],
        "card_data": card_data,
    }


def _size_category(num_examples: int | None) -> str:
    if num_examples is None:
        return "unknown"
    if num_examples < 1_000:
        return "n<1K"
    if num_examples < 10_000:
        return "1K<n<10K"
    if num_examples < 100_000:
        return "10K<n<100K"
    if num_examples < 1_000_000:
        return "100K<n<1M"
    return "n>1M"


def _render_job_stats(stats: JobStats | None) -> str:
    if not stats:
        return "Job statistics could not be collected."

    prompt_total = format_number(stats.prompt_tokens_total) if stats.prompt_tokens_total else "n/a"
    completion_total = format_number(stats.completion_tokens_total) if stats.completion_tokens_total else "n/a"

    rows = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Documents processed | {format_number(stats.document_count)} |",
        f"| Avg. source chars | {stats.mean_doc_len:.2f} |" if stats.mean_doc_len else "",
        f"| Total prompt tokens | {prompt_total} |",
        f"| Total completion tokens | {completion_total} |",
        f"| Mean prompt tokens | {stats.prompt_tokens_mean:.2f} |" if stats.prompt_tokens_mean else "",
        f"| Mean completion tokens | {stats.completion_tokens_mean:.2f} |" if stats.completion_tokens_mean else "",
    ]
    return "\n".join(filter(None, rows))


def format_number(n: int | None) -> str:
    """Format number with human-readable suffix if > 1M."""
    if n is None:
        n = 0
    if n >= 1_000_000_000_000:  # 1 trillion
        trillions = n / 1_000_000_000_000
        return f"{n:,} (~{trillions:.1f}T)"
    elif n >= 1_000_000_000:  # 1 billion
        billions = n / 1_000_000_000
        return f"{n:,} (~{billions:.1f}B)"
    elif n >= 1_000_000:  # 1 million
        millions = n / 1_000_000
        return f"{n:,} (~{millions:.1f}M)"
    return f"{n:,}"


def _format_block(values: Iterable[str], fallback_line: str) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return fallback_line
    return "\n".join(f"- {value}" for value in cleaned)


def _render_template(context: dict[str, str]) -> str:
    template_text = Path(__file__).with_name("dataset_card_template.md").read_text(encoding="utf-8")
    rendered = template_text
    for key, value in context.items():
        rendered = rendered.replace(f"[[{key}]]", value)
    return rendered


def build_and_upload_dataset_card(
    *,
    params: InferenceDatasetCardParams,
    progress_section: str = "",
) -> None:
    """
    Build and upload dataset card.

    Args:
        params: Dataset card parameters, including stats_path.
        progress_section: Optional progress bar string.
    """
    # Load stats only if we're generating the final card (no progress_section)
    # During progress monitoring, we don't want to wait for stats.json
    stats = None
    if not progress_section and params.stats_path:
        stats = load_job_stats(params.stats_path)

    status = "final" if stats else "progress"
    logger.info(
        f"{'Starting dataset card generation' if stats else 'Uploading progress update'} for {params.output_repo_id}"
    )

    if params.stats_path and not stats:
        logger.warning("Warning: Could not load job statistics. Card will have missing data.")

    # Format stats-based values
    fallback = "In progress..." if status == "progress" else "N/A"
    doc_count_text = format_number(stats.document_count) if stats and stats.document_count else fallback
    completion_tokens_text = (
        format_number(stats.completion_tokens_total) if stats and stats.completion_tokens_total else fallback
    )
    size_category = _size_category(stats.document_count) if stats and stats.document_count else "unknown"
    job_stats_table = (
        _render_job_stats(stats)
        if stats
        else (
            "Generation in progress. Final statistics will be available upon completion."
            if status == "progress"
            else "Job statistics could not be collected."
        )
    )

    # Create stats summary line (only show when we have final stats)
    stats_summary = ""  # Empty during progress
    if stats and stats.document_count and stats.completion_tokens_total:
        stats_summary = f"The run produced {doc_count_text} samples and generated {completion_tokens_text} tokens."

    # Fetch source dataset metadata
    source_meta = fetch_source_dataset_metadata(params.input_dataset_name)
    source_dataset_full = (
        f"{params.input_dataset_name}/{params.input_dataset_config}"
        if params.input_dataset_config
        else params.input_dataset_name
    )

    # Extract metadata fields
    raw_languages = source_meta.get("languages")
    languages = ([raw_languages] if isinstance(raw_languages, str) else list(raw_languages or [])) or ["en"]
    license_id = source_meta.get("license") or "other"
    tags = sorted(
        {
            "synthetic",
            *(source_meta.get("tags") or []),
            params.model_name.split("/")[-1],
            params.input_dataset_name.split("/")[-1],
        }
    )

    try:
        hf_user = whoami()["name"]
    except Exception:
        hf_user = "hf_user"

    # Format user prompt info based on whether template is used
    if params.prompt_template:
        user_prompt_info = f"Template `{params.prompt_template}` with content from column `{params.prompt_column}`"
    else:
        user_prompt_info = f"Column `{params.prompt_column}`"

    context = {
        "repo_id": params.output_repo_id,
        "hf_user": hf_user,
        "language_block": _format_block(languages, "- multilingual"),
        "license_id": license_id,
        "tags_block": _format_block(tags, "- dataforge"),
        "size_category": size_category,
        "source_datasets_block": _format_block([source_dataset_full], f"- {source_dataset_full}"),
        "input_dataset_split": params.input_dataset_split or "train",
        "model_name": params.model_name,
        "model_revision": params.model_revision,
        "source_dataset_full": source_dataset_full,
        "model_max_context": str(params.generation_kwargs.get("model_max_context")),
        "temperature": str(params.generation_kwargs.get("temperature")),
        "top_p": str(params.generation_kwargs.get("top_p")),
        "top_k": str(params.generation_kwargs.get("top_k")),
        "max_tokens": str(params.generation_kwargs.get("max_tokens")),
        "spec_config": params.spec_config or "disabled",
        "prompt_column": params.prompt_column,
        "system_prompt": params.system_prompt or "None",
        "user_prompt_info": user_prompt_info,
        "job_stats_table": job_stats_table,
        "progress_section": progress_section,
        "stats_summary": stats_summary,
    }

    content = _render_template(context)

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"Uploading dataset card to {params.output_repo_id}...")
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=params.output_repo_id,
            repo_type="dataset",
        )
        if stats:
            logger.info(f"Successfully uploaded dataset card to {params.output_repo_id}")
        else:
            logger.info(f"Successfully uploaded progress update to {params.output_repo_id}")
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    import typer

    # Create a CLI that just accepts the json args
    def main(args_json: str):
        kwargs = json.loads(args_json)
        build_and_upload_dataset_card(**kwargs)

    typer.run(main)
