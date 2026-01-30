#!/usr/bin/env python3
"""
Analyze benchmark experiment outputs and produce per-experiment CSV files and formatted tables.

- Extract token counts and request stats from inference_logs/stats.json
- Parse throughput metrics from server logs (excludes startup time)
- Compute per-TP (per-GPU) throughput by dividing engine throughput by TP
- Compute derived metrics: gpu_days, node_days, and gpus_for_1b_tokens_per_hour
- Output one table per experiment and save per-experiment CSV files named by experiment name
- Also saves a combined CSV with all experiments

Usage:

# Summarize experiments under the default 'examples/inference/benchmark/results' root and write CSV
python examples/inference/benchmark/analyze_results.py

# Summarize a specific root and set an explicit CSV path
python examples/inference/benchmark/analyze_results.py --root examples/inference/benchmark/results --out-csv examples/inference/benchmark/results/benchmarking_results.csv
"""

import glob
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import pandas as pd
import typer

from datatrove.utils.logging import logger


GPUS_PER_NODE = 8


@dataclass
class PathFields:
    """Parsed experiment configuration fields from a result file path."""

    experiment: str
    model: str
    tp: int | None
    pp: int | None
    dp: int | None
    mns: int | None
    mnbt: int | None
    spec: str
    quant: str
    kv: str


# Patterns indicating OOM errors in vLLM server logs
OOM_PATTERNS = [
    re.compile(r"torch\.OutOfMemoryError.*CUDA out of memory", re.IGNORECASE),
    re.compile(r"ValueError.*No available memory for the cache blocks", re.IGNORECASE),
    re.compile(r"OutOfMemoryError", re.IGNORECASE),
]


def check_oom_in_server_log(server_log_path: Path | None) -> bool:
    """Check if a server log file contains OOM (Out of Memory) errors."""
    if server_log_path is None or not server_log_path.exists():
        return False
    return any(pattern.search(server_log_path.read_text()) for pattern in OOM_PATTERNS)


def parse_server_logs(server_log_path: Path) -> dict[str, float | None] | None:
    """
    Parse vLLM server logs to extract throughput metrics, excluding server startup time.

    Returns:
        Dict with avg_prompt_throughput and avg_generation_throughput (total for engine),
        or None if parsing fails. Divide by TP to get per-GPU throughput.
    """
    if not server_log_path.exists():
        return None

    # Pattern: "Avg prompt throughput: 6949.8 tokens/s, Avg generation throughput: 5108.0 tokens/s"
    pattern = re.compile(
        r"Avg prompt throughput:\s+([\d.]+)\s+tokens/s,\s+Avg generation throughput:\s+([\d.]+)\s+tokens/s"
    )

    prompt_throughputs: list[float] = []
    gen_steady: list[float] = []  # Pure generation phase (prompt=0)
    gen_mixed: list[float] = []  # Mixed phase (prompt>0)

    for line in server_log_path.read_text().splitlines():
        if match := pattern.search(line):
            prompt_thr, gen_thr = float(match.group(1)), float(match.group(2))
            if prompt_thr == 0.0:
                gen_steady.append(gen_thr)
            else:
                prompt_throughputs.append(prompt_thr)
                gen_mixed.append(gen_thr)

    avg_prompt = mean(prompt_throughputs) if prompt_throughputs else None
    avg_gen = mean(gen_steady) if gen_steady else (mean(gen_mixed) if gen_mixed else None)

    if avg_prompt is None and avg_gen is None:
        return None

    return {"avg_prompt_throughput": avg_prompt, "avg_generation_throughput": avg_gen}


def parse_path_fields(file_path: str, root: str = "examples/inference/benchmark/results") -> PathFields:
    """
    Parse experiment configuration from a result file path.

    Expected path structure:
        {root}/{experiment}/model_name/tp{X}-pp{Y}-dp{Z}/mns_{N}/mnbt_{M}/spec_{...}/quant_{...}/kv_{...}/inference_logs/...
    """
    parts = list(Path(file_path).parts)
    root_parts = list(Path(root).parts)

    # Extract experiment name: first directory after root
    experiment = parts[len(root_parts)] if len(parts) > len(root_parts) else ""

    # Find the 'tp' segment and extract tp/pp/dp
    tp, pp, dp, tp_idx = None, None, None, None
    for i, part in enumerate(parts):
        if m := re.match(r"^tp[_-]?(\d+)(?:[_-]?pp[_-]?(\d+))?(?:[_-]?dp[_-]?(\d+))?", part):
            tp = int(m.group(1))
            pp = int(m.group(2)) if m.group(2) else None
            dp = int(m.group(3)) if m.group(3) else None
            tp_idx = i
            break

    # Model is the directory right before the 'tp_*' segment
    model = parts[tp_idx - 1] if tp_idx and tp_idx >= 1 else ""

    def parse_segment(offset: int, prefix: str, default: str | None = None) -> str | int | None:
        """Parse a segment at tp_idx + offset with given prefix."""
        if tp_idx is None or tp_idx + offset >= len(parts):
            return default
        part = parts[tp_idx + offset]
        if part.startswith(prefix):
            suffix = part[len(prefix) :]
            if prefix in ("mns_", "mnbt_"):
                try:
                    return int(suffix)
                except ValueError:
                    return None
            return part
        return default

    return PathFields(
        experiment=experiment,
        model=model,
        tp=tp,
        pp=pp,
        dp=dp,
        mns=parse_segment(1, "mns_"),
        mnbt=parse_segment(2, "mnbt_"),
        spec=parse_segment(3, "spec_", "spec_none") or "spec_none",
        quant=parse_segment(4, "quant_", "quant_none") or "quant_none",
        kv=parse_segment(5, "kv_", "kv_auto") or "kv_auto",
    )


def _get_total_from_stat_field(val: object) -> float | None:
    """Normalize a stats.json field that may be a number or a dict with a 'total' key."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict) and "total" in val:
        t = val["total"]
        if isinstance(t, (int, float)):
            return float(t)
    return None


def parse_stats_json(stats_path: Path) -> dict[str, float | int] | None:
    """
    Parse a merged stats.json to extract token counts and request stats.
    Does NOT extract timing information - use server logs for throughput metrics.
    """
    if not stats_path.exists():
        return None

    data = json.loads(stats_path.read_text())
    if not isinstance(data, list):
        return None

    # Find the "Model call" entry
    entry = next(
        (item for item in data if isinstance(item, dict) and "name" in item and "Model call" in str(item["name"])),
        None,
    )
    if not entry or "stats" not in entry:
        return None

    # Stats fields are optional, default to 0 if missing
    st = entry["stats"]

    def get_stat(key: str) -> float:
        return _get_total_from_stat_field(st[key]) if key in st else 0.0

    return {
        "input_tokens_total": get_stat("prompt_tokens"),
        "output_tokens_total": get_stat("completion_tokens"),
        "successful_requests_total": int(get_stat("successful_requests")),
        "failed_requests_total": int(get_stat("failed_requests")),
    }


def compute_days_for_1b_from_per_gpu(output_tps_per_gpu: float | None) -> tuple[float | None, float | None]:
    """Compute GPU-days and node-days to process 1e9 output tokens using per-GPU throughput."""
    if not output_tps_per_gpu or output_tps_per_gpu <= 0:
        return None, None
    gpu_days = 1_000_000_000.0 / output_tps_per_gpu / 86400.0
    return gpu_days, gpu_days / GPUS_PER_NODE


def compute_gpus_for_1b_per_hour_from_per_gpu(output_tps_per_gpu: float | None) -> float | None:
    """Compute GPUs required to generate 1e9 output tokens in one hour."""
    if not output_tps_per_gpu or output_tps_per_gpu <= 0:
        return None
    return 1_000_000_000.0 / (output_tps_per_gpu * 3600.0)


def analyze(root: str, out_csv: str) -> int:
    files = sorted(glob.glob(os.path.join(root, "**", "inference_logs", "slurm_logs", "*.out"), recursive=True))
    rows: list[dict[str, object]] = []

    for path in files:
        fields = parse_path_fields(path, root)

        # Create row from path fields + additional metrics (default None)
        row: dict[str, object] = {
            **asdict(fields),
            "util": None,
            "gpus_for_1b_tokens_per_hour": None,
            "node_days_to_process_1b_tokens": None,
            "gpu_days_to_process_1b_tokens": None,
            "input_tps_per_gpu": None,
            "output_tps_per_gpu": None,
            "tokens_input_per_sec": None,
            "tokens_output_per_sec": None,
            "prompt_tokens_total": None,
            "completion_tokens_total": None,
            "successful_requests": None,
            "failed_requests": None,
            "path": path,
            "comment": "",
        }

        stats_dir = Path(path).parent.parent  # .../inference_logs

        # Find server log file (try node-specific pattern first, then fallback)
        server_logs_dir = stats_dir / "server_logs"
        candidates = (
            (
                sorted(server_logs_dir.glob("server_rank_*_node_*.log"))
                or sorted(server_logs_dir.glob("server_rank_*.log"))
            )
            if server_logs_dir.exists()
            else []
        )
        server_log_path = candidates[0] if candidates else None

        # Check for OOM errors BEFORE stats.json - OOM failures prevent stats.json creation
        if check_oom_in_server_log(server_log_path):
            row["comment"] = "OOM"
            rows.append(row)
            continue

        stats_info = parse_stats_json(stats_dir / "stats.json")
        if stats_info is None:
            row["comment"] = "stats.json not found"
            rows.append(row)
            continue

        row["prompt_tokens_total"] = int(stats_info["input_tokens_total"])
        row["completion_tokens_total"] = int(stats_info["output_tokens_total"])
        row["successful_requests"] = stats_info["successful_requests_total"]
        row["failed_requests"] = stats_info["failed_requests_total"]

        server_metrics = parse_server_logs(server_log_path) if server_log_path else None
        if server_metrics is None:
            row["comment"] = "server logs not found or unparseable"
            rows.append(row)
            continue

        avg_prompt_thr = server_metrics["avg_prompt_throughput"]
        avg_gen_thr = server_metrics["avg_generation_throughput"]
        if avg_prompt_thr is None or avg_gen_thr is None:
            row["comment"] = "throughput metrics not found in server logs"
            rows.append(row)
            continue

        # Total GPUs contributing to this throughput = TP * PP (normalize to 1 if None/0)
        gpu_divisor = (fields.tp or 1) * (fields.pp or 1)

        row["tokens_input_per_sec"] = avg_prompt_thr
        row["input_tps_per_gpu"] = avg_prompt_thr / gpu_divisor
        row["tokens_output_per_sec"] = avg_gen_thr
        row["output_tps_per_gpu"] = avg_gen_thr / gpu_divisor

        gpu_days, node_days = compute_days_for_1b_from_per_gpu(row["output_tps_per_gpu"])
        row["gpu_days_to_process_1b_tokens"] = gpu_days
        row["node_days_to_process_1b_tokens"] = node_days
        row["gpus_for_1b_tokens_per_hour"] = compute_gpus_for_1b_per_hour_from_per_gpu(row["output_tps_per_gpu"])

        if row["failed_requests"] and row["failed_requests"] > 0:
            row["comment"] = f"failed_requests={row['failed_requests']}"

        rows.append(row)

    # Column order:
    # - experiment name first
    # - config columns (including quant_config and kv_cache_config after spec_config)
    # - per-sec cluster rates and per-TP rates
    # - gpu/node days (computed from per-TP output tokens)
    # - then: prompt_tokens_total, completion_tokens_total, successful_requests (immediately before path)
    # - path
    fieldnames = [
        "experiment",
        "model",
        "tp",
        "pp",
        "dp",
        "mns",
        "mnbt",
        "util",
        "spec",
        "quant",
        "kv",
        "gpus_for_1b_tokens_per_hour",
        "node_days_to_process_1b_tokens",
        "gpu_days_to_process_1b_tokens",
        "input_tps_per_gpu",
        "output_tps_per_gpu",
        "tokens_input_per_sec",
        "tokens_output_per_sec",
        "prompt_tokens_total",
        "completion_tokens_total",
        "successful_requests",
        "path",
        "comment",
    ]

    df = pd.DataFrame(rows, columns=fieldnames)
    # Ensure integer columns are properly typed
    for col in ["tp", "pp", "dp", "mns", "mnbt"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Columns used for deduplication and metrics for sorting
    dedup_cols = ["experiment", "model", "tp", "pp", "dp", "mns", "mnbt", "spec", "quant", "kv"]
    metric_cols = [
        "node_days_to_process_1b_tokens",
        "gpu_days_to_process_1b_tokens",
        "input_tps_per_gpu",
        "output_tps_per_gpu",
    ]

    # Deduplicate: keep row with most metrics; if tied, keep highest output_tps_per_gpu
    def deduplicate(df_in: pd.DataFrame) -> pd.DataFrame:
        df_work = df_in.copy()
        df_work["__metric_count__"] = df_work[metric_cols].notna().sum(axis=1)
        df_work["__out_tok_tp__"] = df_work["output_tps_per_gpu"].fillna(-1)
        return (
            df_work.sort_values(
                by=dedup_cols + ["__metric_count__", "__out_tok_tp__"],
                ascending=[True] * len(dedup_cols) + [False, False],
            )
            .drop_duplicates(subset=dedup_cols, keep="first")
            .drop(columns=["__metric_count__", "__out_tok_tp__"])
        )

    df_dedup = deduplicate(df)

    # Config columns for table display (may be constant or varying per experiment)
    config_columns = [
        ("model", "model", "left"),
        ("tp", "tp", "right"),
        ("pp", "pp", "right"),
        ("dp", "dp", "right"),
        ("mns", "mns", "right"),
        ("mnbt", "mnbt", "right"),
        ("spec", "spec", "left"),
        ("quant", "quant", "left"),
        ("kv", "kv", "left"),
    ]
    # Metric columns always shown in table
    display_metric_columns = [
        ("gpus/1b/h", "gpus_for_1b_tokens_per_hour", "right"),
        ("in tps/gpu", "input_tps_per_gpu", "right"),
        ("out tps/gpu", "output_tps_per_gpu", "right"),
        ("comment", "comment", "left"),
    ]

    def fmt(val: object, col_key: str) -> str:
        """Format a value for table display."""
        if val is None or pd.isna(val):
            return ""
        if isinstance(val, (int, float)):
            if col_key in ("node_days_to_process_1b_tokens", "gpu_days_to_process_1b_tokens"):
                return f"{float(val):.3f}"
            if col_key in (
                "input_tps_per_gpu",
                "output_tps_per_gpu",
                "tp",
                "pp",
                "dp",
                "mns",
                "mnbt",
                "gpus_for_1b_tokens_per_hour",
            ):
                return f"{int(round(float(val)))}"
            return f"{float(val):.1f}"
        # Strip prefixes from config columns for narrower display
        str_val = str(val)
        for prefix in [("spec", "spec_"), ("quant", "quant_"), ("kv", "kv_")]:
            if col_key == prefix[0] and str_val.startswith(prefix[1]):
                return str_val[len(prefix[1]) :]
        return str_val

    def print_table(df_exp: pd.DataFrame, experiment_name: str) -> None:
        """Print a formatted table for a single experiment with dynamic columns."""
        # Identify which config columns have varying vs constant values
        varying_configs: list[tuple[str, str, str]] = []
        constant_configs: list[tuple[str, str]] = []

        for label, col_key, align in config_columns:
            unique_vals = df_exp[col_key].dropna().unique()
            if len(unique_vals) <= 1:
                val = fmt(unique_vals[0], col_key) if len(unique_vals) == 1 else ""
                if val:
                    constant_configs.append((label, val))
            else:
                varying_configs.append((label, col_key, align))

        dynamic_columns = varying_configs + display_metric_columns
        header_labels = [h for h, _, _ in dynamic_columns]
        aligns = [align for _, _, align in dynamic_columns]

        # Build data rows
        data_rows = [[fmt(row[col_key], col_key) for _, col_key, _ in dynamic_columns] for _, row in df_exp.iterrows()]

        # Compute column widths
        widths = [max(len(h), max((len(r[i]) for r in data_rows), default=0)) for i, h in enumerate(header_labels)]

        def pad(cell: str, width: int, align: str) -> str:
            return cell.rjust(width) if align == "right" else cell.ljust(width)

        # Print experiment header
        print(f"\n{'=' * 60}")
        header_parts = [f"Experiment: {experiment_name}"]
        if constant_configs:
            header_parts.append(f"[{', '.join(f'{label}={val}' for label, val in constant_configs)}]")
        print(" ".join(header_parts))
        print(f"{'=' * 60}")

        # Print table
        print("| " + " | ".join(pad(h, widths[i], "left") for i, h in enumerate(header_labels)) + " |")
        print("| " + " | ".join("-" * w for w in widths) + " |")
        for r in data_rows:
            print("| " + " | ".join(pad(r[i], widths[i], aligns[i]) for i in range(len(r))) + " |")

    # Process each experiment
    out_csv_dir = Path(out_csv).parent
    total_rows = 0

    for experiment_name in sorted(df_dedup["experiment"].unique()):
        df_exp = df_dedup[df_dedup["experiment"] == experiment_name]
        print_table(df_exp, experiment_name)

        # Save CSV for this experiment
        exp_csv_path = out_csv_dir / f"{experiment_name}.csv"
        df_exp.to_csv(exp_csv_path, index=False, float_format="%.6f")
        logger.info(f"Wrote {len(df_exp)} rows to {exp_csv_path}")
        total_rows += len(df_exp)

    # Save combined CSV
    df_dedup.to_csv(out_csv, index=False, float_format="%.6f")
    logger.info(
        f"Wrote {total_rows} total rows across {len(df_dedup['experiment'].unique())} experiments to {out_csv}"
    )

    return total_rows


def main(
    root: str = "examples/inference/benchmark/results",
    out_csv: str = "examples/inference/benchmark/results/benchmarking_results.csv",
) -> None:
    analyze(root, out_csv)


if __name__ == "__main__":
    typer.run(main)
