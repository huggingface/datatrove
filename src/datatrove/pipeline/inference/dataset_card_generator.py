import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Iterable

import yaml
from huggingface_hub import dataset_info, hf_hub_download, upload_file, whoami

from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger


_PROMPT_BLOCK_PATTERN = re.compile(
    r"<summary><b>(\w+)</b> prompt</summary>\s*\n\s*<(?P<tag>pre|div)[^>]*>(?P<body>.*?)</(?P=tag)>",
    re.DOTALL,
)


@dataclass
class InferenceDatasetCardParams:
    """Parameters required for generating a dataset card."""

    output_repo_id: str
    input_dataset_name: str
    input_dataset_split: str
    input_dataset_config: str | None
    prompt_column: str
    prompt_template: str | None
    prompt_template_name: str  # "default" or a named config like "math", "faq", etc.
    system_prompt: str | None
    model_name: str
    model_revision: str
    generation_kwargs: dict[str, Any]
    spec_config: str | None
    stats_path: str


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
    """Format number with human-readable suffix if > 1M.

    Uses ≈ instead of ~ to avoid markdown strikethrough interpretation.
    """
    if n is None:
        n = 0
    if n >= 1_000_000_000_000:  # 1 trillion
        trillions = n / 1_000_000_000_000
        return f"{n:,} (≈{trillions:.1f}T)"
    elif n >= 1_000_000_000:  # 1 billion
        billions = n / 1_000_000_000
        return f"{n:,} (≈{billions:.1f}B)"
    elif n >= 1_000_000:  # 1 million
        millions = n / 1_000_000
        return f"{n:,} (≈{millions:.1f}M)"
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


def _build_config_entry(config_name: str, split: str, data_path: str) -> dict[str, Any]:
    """Build a single HF dataset config entry."""
    return {
        "config_name": config_name,
        "data_files": [{"split": split, "path": data_path}],
    }


def _data_path_for_config(config_name: str) -> str:
    return f"{config_name}/**/*.parquet"


def _fetch_existing_configs(repo_id: str) -> list[dict[str, Any]]:
    """Fetch existing configs from the HF repo's README.md YAML frontmatter.

    Returns an empty list if the README doesn't exist or has no configs.
    """
    try:
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="dataset")
        content = Path(readme_path).read_text(encoding="utf-8")
        # Parse YAML frontmatter between --- markers
        if content.startswith("---"):
            end = content.index("---", 3)
            frontmatter = yaml.safe_load(content[3:end])
            return frontmatter.get("configs", []) if frontmatter else []
    except Exception:
        pass
    return []


def _merge_configs(
    existing_configs: list[dict[str, Any]],
    new_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge a new config into existing configs, replacing any with the same config_name."""
    configs = [c for c in existing_configs if c["config_name"] != new_config["config_name"]]
    configs.append(new_config)
    return sorted(configs, key=lambda c: c["config_name"])


def _add_all_config(configs: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Add an 'all' config that unions all named configs' data files.

    Only added when there are multiple non-default configs.
    Uses a single data_files entry with a list of paths to avoid
    duplicate split names (which HuggingFace rejects).
    """
    named = [c for c in configs if c["config_name"] not in ("default", "all")]
    if len(named) < 2:
        return configs

    # Collect all data paths from named configs into a single split entry
    all_paths = [df["path"] for cfg in named for df in cfg["data_files"]]

    all_config = {
        "config_name": "all",
        "data_files": [{"split": split, "path": all_paths}],
    }

    # Remove any existing "all" config, then prepend
    result = [c for c in configs if c["config_name"] != "all"]
    result.insert(0, all_config)
    return result


def _render_configs_block(configs: list[dict[str, Any]], split: str) -> str:
    """Render the configs and train-eval-index YAML block for the dataset card frontmatter."""
    lines = ["configs:"]
    for cfg in configs:
        lines.append(f"- config_name: {cfg['config_name']}")
        lines.append("  data_files:")
        for df in cfg["data_files"]:
            lines.append(f"  - split: {df['split']}")
            path = df["path"]
            if isinstance(path, list):
                lines.append("    path:")
                for p in path:
                    lines.append(f"    - {p}")
            else:
                lines.append(f"    path: {path}")

    # train-eval-index references the "all" config if present, else the first config
    default_config = next(
        (c["config_name"] for c in configs if c["config_name"] == "all"),
        configs[0]["config_name"] if configs else "default",
    )
    lines.extend(
        [
            "train-eval-index:",
            f"- config: {default_config}",
            "  task: text-generation",
            "  task_id: language-modeling",
            "  splits:",
            f"    train_split: {split}",
            "    eval_split:",
            "  col_mapping:",
            "    text: text",
        ]
    )
    return "\n".join(lines)


def _parse_existing_prompts(repo_id: str) -> dict[str, str]:
    """Parse existing per-config prompt templates from the README.

    Looks for <pre> blocks inside <details> tags with summary like:
        <summary><b>faq</b> prompt</summary>
    followed by a <pre> block containing the template text.

    Returns:
        Mapping of config_name -> raw prompt template string.
    """
    try:
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="dataset")
        content = Path(readme_path).read_text(encoding="utf-8")
    except Exception:
        return {}

    return _extract_prompt_templates(content)


def _decode_prompt_html_content(content: str) -> str:
    """Decode prompt HTML body back into raw template text."""
    with_newlines = re.sub(r"<br\s*/?>", "\n", content, flags=re.IGNORECASE)
    return unescape(with_newlines).strip()


def _extract_prompt_templates(content: str) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for match in _PROMPT_BLOCK_PATTERN.finditer(content):
        prompts[match.group(1)] = _decode_prompt_html_content(match.group("body"))
    return prompts


def _render_prompt_pre(template: str) -> str:
    """Render a prompt template in a word-wrapping <pre> block.

    Uses white-space:pre-wrap so actual newlines in the template are preserved
    but long lines wrap instead of showing a horizontal scrollbar.

    Newlines are encoded as <br/> so the markdown source contains no raw
    newline inside the tag body, which prevents HF markdown from leaking text
    out of the block.
    """
    safe = template.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")
    return f'<pre style="white-space: pre-wrap;">{safe}</pre>'


def _render_user_prompt_info(
    current_config: str,
    current_template: str | None,
    prompt_column: str,
    existing_prompts: dict[str, str],
) -> str:
    """Render the user prompt section, merging current config with existing ones.

    For multiple named configs, renders collapsible details per config, indented
    under the "User prompts" bullet. For a single "default" config, renders a
    simple inline description.
    """
    all_prompts = dict(existing_prompts)
    if current_template:
        all_prompts[current_config] = current_template

    if not all_prompts or (len(all_prompts) == 1 and current_config == "default"):
        if current_template:
            pre = _render_prompt_pre(current_template)
            return (
                f" * User prompt: Template with content from column `{prompt_column}`\n\n"
                f"   <details>\n   <summary>Prompt template</summary>\n\n"
                f"   {pre}\n\n"
                f"   </details>"
            )
        return f" * User prompt: Column `{prompt_column}`"

    return _render_user_prompts_block(prompt_column, all_prompts)


def _render_user_prompts_block(prompt_column: str, prompts: dict[str, str]) -> str:
    """Render a stable, indentation-safe multi-config user prompts section."""
    lines = [f" * User prompts (from column `{prompt_column}`):", "   "]
    for name in sorted(prompts):
        pre = _render_prompt_pre(prompts[name])
        lines.append("   <details>")
        lines.append(f"   <summary><b>{name}</b> prompt</summary>")
        lines.append("   ")
        lines.append(f"   {pre}")
        lines.append("   ")
        lines.append("   </details>")
        lines.append("   ")
    return "\n".join(lines).rstrip()


def patch_readme_prompt(readme_content: str, config_name: str, template: str) -> str:
    """Upsert a config prompt by re-rendering the full user prompts section."""
    section_header_match = re.search(
        r"^\s*\* User prompts? \(from column `([^`]+)`\):\s*$", readme_content, re.MULTILINE
    )
    if not section_header_match:
        return readme_content

    prompt_column = section_header_match.group(1)
    section_start = section_header_match.start()

    next_section_match = re.search(r"^\s*##\s+", readme_content[section_header_match.end() :], re.MULTILINE)
    if not next_section_match:
        section_end = len(readme_content)
    else:
        section_end = section_header_match.end() + next_section_match.start()

    section_text = readme_content[section_start:section_end]
    existing_prompts = _extract_prompt_templates(section_text)
    existing_prompts[config_name] = template
    rebuilt_section = _render_user_prompts_block(prompt_column, existing_prompts)

    trailing = readme_content[section_end:].lstrip("\n")
    return readme_content[:section_start] + rebuilt_section + "\n\n" + trailing


def patch_readme_configs(readme_content: str, repo_id: str, config_name: str, split: str) -> str:
    """Patch the YAML frontmatter configs and load_dataset example for a new config.

    Merges the new config into existing ones, adds the "all" config if needed,
    and regenerates the configs YAML block and load_dataset example.
    """
    # Parse existing configs from frontmatter
    existing_configs: list[dict[str, Any]] = []
    if readme_content.startswith("---"):
        end = readme_content.index("---", 3)
        frontmatter = yaml.safe_load(readme_content[3:end])
        existing_configs = (frontmatter or {}).get("configs", [])

    # Check if this config already exists
    if any(c["config_name"] == config_name for c in existing_configs):
        return readme_content

    # Build and merge the new config
    data_path = _data_path_for_config(config_name)
    new_config = _build_config_entry(config_name, split, data_path)
    merged = _merge_configs(existing_configs, new_config)
    merged = _add_all_config(merged, split)

    # Replace the configs + train-eval-index block in the YAML frontmatter
    new_block = _render_configs_block(merged, split)
    # Match from "configs:" to the end of the frontmatter (the closing ---)
    readme_content = re.sub(
        r"configs:.*?(?=\n---)",
        new_block,
        readme_content,
        count=1,
        flags=re.DOTALL,
    )

    # Replace the load_dataset example
    new_example = _render_load_dataset_example(repo_id, merged)
    readme_content = re.sub(
        r"You can load the dataset using\n```python\n.*?```",
        new_example,
        readme_content,
        count=1,
        flags=re.DOTALL,
    )

    return readme_content


def _render_load_dataset_example(repo_id: str, configs: list[dict[str, Any]]) -> str:
    """Render the load_dataset usage example for the dataset card."""
    has_named_configs = any(c["config_name"] not in ("default",) for c in configs)
    lines = ["You can load the dataset using", "```python", "from datasets import load_dataset", ""]
    if has_named_configs:
        # Show "all" first if present, then individual configs
        for cfg in configs:
            name = cfg["config_name"]
            var_name = f"ds_{name}" if name != "all" else "ds"
            comment = "  # all subsets combined" if name == "all" else ""
            lines.append(f'{var_name} = load_dataset("{repo_id}", "{name}"){comment}')
    else:
        lines.append(f'ds = load_dataset("{repo_id}")')
    lines.append("```")
    return "\n".join(lines)


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
        stats = load_job_stats(Path(params.stats_path))

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

    # Format user prompt info, merging with any existing configs' prompts in the README
    existing_prompts = _parse_existing_prompts(params.output_repo_id)
    user_prompt_info = _render_user_prompt_info(
        params.prompt_template_name, params.prompt_template, params.prompt_column, existing_prompts
    )

    # Build HF dataset configs: merge this job's config with any existing ones in the repo
    split = params.input_dataset_split or "train"
    config_name = params.prompt_template_name
    data_path = _data_path_for_config(config_name)
    new_config = _build_config_entry(config_name, split, data_path)
    existing_configs = _fetch_existing_configs(params.output_repo_id)
    merged_configs = _merge_configs(existing_configs, new_config)
    merged_configs = _add_all_config(merged_configs, split)

    context = {
        "repo_id": params.output_repo_id,
        "hf_user": hf_user,
        "language_block": _format_block(languages, "- multilingual"),
        "license_id": license_id,
        "tags_block": _format_block(tags, "- dataforge"),
        "size_category": size_category,
        "source_datasets_block": _format_block([source_dataset_full], f"- {source_dataset_full}"),
        "source_dataset_info": (
            f"`{params.input_dataset_config}` config, `{split}` split"
            if params.input_dataset_config
            else f"`{split}` split"
        ),
        "model_name": params.model_name,
        "model_revision": params.model_revision,
        "source_dataset_full": source_dataset_full,
        "source_dataset_link": params.input_dataset_name,
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
        "configs_block": _render_configs_block(merged_configs, split),
        "load_dataset_example": _render_load_dataset_example(params.output_repo_id, merged_configs),
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
