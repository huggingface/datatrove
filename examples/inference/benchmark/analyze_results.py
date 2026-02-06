#!/usr/bin/env python3
"""
Analyze benchmark experiment outputs and produce per-experiment CSV files and formatted tables.

- Extract token counts and request stats from inference_logs/stats.json
- Parse throughput metrics from server logs (excludes startup time)
- Compute per-TP (per-GPU) throughput by dividing engine throughput by TP
- Compute derived metrics: gpu_days, node_days, and gpus_for_1b_tokens_per_hour
- Output one table per experiment and save per-experiment CSV files named by experiment name to the root directory
- Also saves a combined CSV with all experiments to the root directory

Usage:

# Summarize experiments under the default 'examples/inference/benchmark/results' root and write CSV
python examples/inference/benchmark/analyze_results.py

# Summarize a specific root and write CSVs to that root
python examples/inference/benchmark/analyze_results.py --root examples/inference/benchmark/results

# Control parallelism (default: -1 uses all CPUs, use 1 for sequential processing)
python examples/inference/benchmark/analyze_results.py --n-jobs 8
"""

import glob
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import pandas as pd
import typer
from joblib import Parallel, delayed
from tqdm import tqdm

from datatrove.utils.logging import logger


# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import detect_failure_reason


GPUS_PER_NODE = 8


@dataclass
class PathFields:
    """Parsed experiment configuration fields from a result file path."""

    experiment: str
    prompt_template_name: str
    model: str
    tp: int | None
    pp: int | None
    dp: int | None
    mns: int | None
    mnbt: int | None
    gmu: int | None  # GPU memory utilization as percentage (e.g., 90 for 0.9)
    bs: int | None  # Block size
    kvc: str  # KV cache dtype
    spec: str
    quant: str


def parse_server_logs(server_log_path: Path) -> dict[str, float | None] | None:
    """
    Parse vLLM server logs to extract throughput metrics, excluding server startup time.

    Returns:
        Dict with avg_prompt_throughput, avg_generation_throughput (total for engine),
        avg_gpu_kvc_usage, avg_prefix_cache_hit_rate, running_reqs, and waiting_reqs,
        or None if parsing fails. Divide throughputs by TP to get per-GPU throughput.
    """
    if not server_log_path.exists():
        return None

    # Pattern to extract throughput and cache metrics from log lines like:
    # "Avg prompt throughput: 6949.8 tokens/s, Avg generation throughput: 5108.0 tokens/s, Running: 2005 reqs, Waiting: 2995 reqs, GPU KV cache usage: 67.7%, Prefix cache hit rate: 4.4%"
    throughput_pattern = re.compile(
        r"Avg prompt throughput:\s+([\d.]+)\s+tokens/s,\s+Avg generation throughput:\s+([\d.]+)\s+tokens/s"
    )
    running_pattern = re.compile(r"Running:\s+(\d+)\s+reqs")
    waiting_pattern = re.compile(r"Waiting:\s+(\d+)\s+reqs")
    kv_cache_pattern = re.compile(r"GPU KV cache usage:\s+([\d.]+)%")
    prefix_cache_pattern = re.compile(r"Prefix cache hit rate:\s+([\d.]+)%")

    prompt_throughputs: list[float] = []
    gen_steady: list[float] = []  # Pure generation phase (prompt=0)
    gen_mixed: list[float] = []  # Mixed phase (prompt>0)
    running_reqs: list[int] = []
    waiting_reqs: list[int] = []
    kv_cache_usages: list[float] = []
    prefix_cache_rates: list[float] = []

    for line in server_log_path.read_text(errors="replace").splitlines():
        if match := throughput_pattern.search(line):
            prompt_thr, gen_thr = float(match.group(1)), float(match.group(2))
            if prompt_thr == 0.0:
                gen_steady.append(gen_thr)
            else:
                prompt_throughputs.append(prompt_thr)
                gen_mixed.append(gen_thr)

            # Extract request counts and cache metrics from the same line
            if running_match := running_pattern.search(line):
                running_reqs.append(int(running_match.group(1)))
            if waiting_match := waiting_pattern.search(line):
                waiting_reqs.append(int(waiting_match.group(1)))
            if kv_match := kv_cache_pattern.search(line):
                kv_cache_usages.append(float(kv_match.group(1)))
            if prefix_match := prefix_cache_pattern.search(line):
                prefix_cache_rates.append(float(prefix_match.group(1)))

    avg_prompt = mean(prompt_throughputs) if prompt_throughputs else None
    avg_gen = mean(gen_steady) if gen_steady else (mean(gen_mixed) if gen_mixed else None)
    avg_running = mean(running_reqs) if running_reqs else None
    avg_waiting = mean(waiting_reqs) if waiting_reqs else None
    avg_kv_cache = mean(kv_cache_usages) if kv_cache_usages else None
    avg_prefix_cache = mean(prefix_cache_rates) if prefix_cache_rates else None

    if avg_prompt is None and avg_gen is None:
        return None

    return {
        "avg_prompt_throughput": avg_prompt,
        "avg_generation_throughput": avg_gen,
        "avg_running_reqs": avg_running,
        "avg_waiting_reqs": avg_waiting,
        "avg_gpu_kvc_usage": avg_kv_cache,
        "avg_prefix_cache_hit_rate": avg_prefix_cache,
    }


def parse_path_fields(file_path: str, root: str = "examples/inference/benchmark/results") -> PathFields:
    """
    Parse experiment configuration from a result file path.

    Expected path structure:
        {root}/{experiment}/{prompt}/{model}/tp{X}-pp{Y}-dp{Z}/mns_{N}/mnbt_{M}/gmu_{P}/bs_{B}/kvc_{...}/spec_{...}/quant_{...}/inference_logs/...
    """
    parts = list(Path(file_path).parts)
    root_parts = list(Path(root).parts)

    # Extract experiment name: first directory after root
    experiment = parts[len(root_parts)] if len(parts) > len(root_parts) else ""

    # Extract prompt_template_name: second directory after root
    prompt_template_name = parts[len(root_parts) + 1] if len(parts) > len(root_parts) + 1 else "default"

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

    def parse_int_segment(offset: int, prefix: str) -> int | None:
        """Parse an integer segment at tp_idx + offset with given prefix."""
        if tp_idx is None or tp_idx + offset >= len(parts):
            return None
        part = parts[tp_idx + offset]
        if part.startswith(prefix):
            try:
                return int(part[len(prefix) :])
            except ValueError:
                return None
        return None

    def parse_str_segment(offset: int, prefix: str, default: str) -> str:
        """Parse a string segment at tp_idx + offset with given prefix."""
        if tp_idx is None or tp_idx + offset >= len(parts):
            return default
        part = parts[tp_idx + offset]
        return part if part.startswith(prefix) else default

    return PathFields(
        experiment=experiment,
        prompt_template_name=prompt_template_name,
        model=model,
        tp=tp,
        pp=pp,
        dp=dp,
        mns=parse_int_segment(1, "mns_"),
        mnbt=parse_int_segment(2, "mnbt_"),
        gmu=parse_int_segment(3, "gmu_"),
        bs=parse_int_segment(4, "bs_"),
        kvc=parse_str_segment(5, "kvc_", "kvc_auto"),
        spec=parse_str_segment(6, "spec_", "spec_none"),
        quant=parse_str_segment(7, "quant_", "quant_none"),
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


def _get_mean_from_stat_field(val: object) -> float | None:
    """Extract the mean from a stats.json field that may be a dict with a 'mean' key."""
    if val is None:
        return None
    if isinstance(val, dict) and "mean" in val:
        m = val["mean"]
        if isinstance(m, (int, float)):
            return float(m)
    return None


def parse_stats_json(stats_path: Path) -> dict[str, float | int] | None:
    """
    Parse a merged stats.json to extract token counts, request stats, and total time.
    """
    if not stats_path.exists():
        return None

    data = json.loads(stats_path.read_text())
    if not isinstance(data, list):
        return None

    # Calculate total e2e time by summing time_stats.global_mean (or total) from all entries
    e2e_time = 0.0
    for item in data:
        if isinstance(item, dict) and "time_stats" in item:
            ts = item["time_stats"]
            if isinstance(ts, dict):
                # Prefer global_mean for merged stats, fallback to total
                e2e_time += ts.get("global_mean", ts.get("total", 0.0))

    # Find the "Model call" entry for token counts and timing stats
    entry = next(
        (item for item in data if isinstance(item, dict) and "name" in item and "Model call" in str(item["name"])),
        None,
    )
    if not entry or "stats" not in entry:
        return None

    st = entry["stats"]

    def get_total(key: str) -> float:
        return _get_total_from_stat_field(st[key]) if key in st else 0.0

    def get_mean(key: str) -> float | None:
        return _get_mean_from_stat_field(st[key]) if key in st else None

    # Extract inference_time if available (saved as TimingStats under stats)
    inference_time = get_total("inference_time") if "inference_time" in st else None

    return {
        "input_tokens_total": get_total("prompt_tokens"),
        "output_tokens_total": get_total("completion_tokens"),
        "input_tokens_mean": get_mean("prompt_tokens"),
        "output_tokens_mean": get_mean("completion_tokens"),
        "successful_requests_total": int(get_total("successful_requests")),
        "failed_requests_total": int(get_total("failed_requests")),
        "e2e_time": e2e_time,
        "inference_time": inference_time,
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


def process_single_file(path: str, root: str) -> dict[str, object]:
    """Process a single log file and return a row dict with metrics."""
    fields = parse_path_fields(path, root)

    # Create row from path fields + additional metrics (default None)
    row: dict[str, object] = {
        **asdict(fields),
        "gpus_for_1b_tokens_per_hour": None,
        "node_days_to_process_1b_tokens": None,
        "gpu_days_to_process_1b_tokens": None,
        "input_tps_per_gpu": None,
        "output_tps_per_gpu": None,
        "tokens_input_per_sec": None,
        "tokens_output_per_sec": None,
        "avg_running_reqs": None,
        "avg_waiting_reqs": None,
        "avg_gpu_kvc_usage": None,
        "avg_prefix_cache_hit_rate": None,
        "prompt_tokens_total": None,
        "completion_tokens_total": None,
        "mean_input_tokens": None,
        "mean_output_tokens": None,
        "successful_requests": None,
        "failed_requests": None,
        "e2e_time": None,
        "inference_time": None,
        "e2e_per_1k_per_gpu": None,
        "inference_per_1k_per_gpu": None,
        "path": path,
        "comment": "",
    }
    stats_dir = Path(path).parent.parent  # .../inference_logs

    # Find server log file (try node-specific pattern first, then fallback)
    server_logs_dir = stats_dir / "server_logs"
    candidates = (
        (sorted(server_logs_dir.glob("server_rank_*_node_*.log")) or sorted(server_logs_dir.glob("server_rank_*.log")))
        if server_logs_dir.exists()
        else []
    )
    server_log_path = candidates[0] if candidates else None

    # Check for failure patterns BEFORE stats.json - failures prevent stats.json creation
    slurm_log_path = Path(path)
    failure_reason = detect_failure_reason(slurm_log_path) or detect_failure_reason(server_log_path)
    if failure_reason:
        row["comment"] = failure_reason
        return row

    stats_info = parse_stats_json(stats_dir / "stats.json")
    if stats_info is None:
        row["comment"] = "stats.json not found"
        return row

    row["prompt_tokens_total"] = int(stats_info["input_tokens_total"])
    row["completion_tokens_total"] = int(stats_info["output_tokens_total"])
    row["mean_input_tokens"] = stats_info["input_tokens_mean"]
    row["mean_output_tokens"] = stats_info["output_tokens_mean"]
    row["successful_requests"] = stats_info["successful_requests_total"]
    row["failed_requests"] = stats_info["failed_requests_total"]
    row["e2e_time"] = stats_info["e2e_time"]
    row["inference_time"] = stats_info["inference_time"]

    # Total GPUs contributing to this throughput = TP * PP (normalize to 1 if None/0)
    gpu_divisor = (fields.tp or 1) * (fields.pp or 1)

    # GPU-time to process 1000 examples (e2e_time * num_gpus / num_requests * 1000)
    if stats_info["successful_requests_total"] > 0 and stats_info["e2e_time"] > 0:
        row["e2e_per_1k_per_gpu"] = (
            stats_info["e2e_time"] / stats_info["successful_requests_total"] * 1000 * gpu_divisor
        )

    # Calculate throughput from inference_time only (excludes server startup)
    inference_time = stats_info["inference_time"]
    if inference_time and inference_time > 0:
        row["tokens_input_per_sec"] = stats_info["input_tokens_total"] / inference_time
        row["input_tps_per_gpu"] = stats_info["input_tokens_total"] / inference_time / gpu_divisor
        row["tokens_output_per_sec"] = stats_info["output_tokens_total"] / inference_time
        row["output_tps_per_gpu"] = stats_info["output_tokens_total"] / inference_time / gpu_divisor
        if stats_info["successful_requests_total"] > 0:
            row["inference_per_1k_per_gpu"] = (
                inference_time / stats_info["successful_requests_total"] * 1000 * gpu_divisor
            )

    # Parse server logs for additional metrics (kv cache, prefix cache, etc.)
    server_metrics = parse_server_logs(server_log_path) if server_log_path else None
    if server_metrics:
        row["avg_running_reqs"] = server_metrics["avg_running_reqs"]
        row["avg_waiting_reqs"] = server_metrics["avg_waiting_reqs"]
        row["avg_gpu_kvc_usage"] = server_metrics["avg_gpu_kvc_usage"]
        row["avg_prefix_cache_hit_rate"] = server_metrics["avg_prefix_cache_hit_rate"]

    gpu_days, node_days = compute_days_for_1b_from_per_gpu(row["output_tps_per_gpu"])
    row["gpu_days_to_process_1b_tokens"] = gpu_days
    row["node_days_to_process_1b_tokens"] = node_days
    row["gpus_for_1b_tokens_per_hour"] = compute_gpus_for_1b_per_hour_from_per_gpu(row["output_tps_per_gpu"])

    if row["failed_requests"] and row["failed_requests"] > 0:
        row["comment"] = f"failed_requests={row['failed_requests']}"

    return row


def analyze(root: str, n_jobs: int = -1) -> int:
    """Analyze benchmark results with parallel log processing.

    Args:
        root: Root directory containing experiment results.
        n_jobs: Number of parallel jobs (-1 for all CPUs).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        message = f"Root directory does not exist: {root_path}"
        logger.error(message)
        raise FileNotFoundError(message)
    if not any(root_path.iterdir()):
        message = f"Root directory is empty: {root_path}"
        logger.error(message)
        raise ValueError(message)

    files = sorted(glob.glob(os.path.join(root, "**", "inference_logs", "slurm_logs", "*.out"), recursive=True))

    # Process files in parallel with progress bar (threading backend for I/O-bound work)
    rows: list[dict[str, object]] = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_single_file)(path, root) for path in tqdm(files, desc="Processing logs")
    )

    # Config columns used for deduplication (experiment + all config params)
    config_cols = [
        "experiment",
        "prompt_template_name",
        "model",
        "tp",
        "pp",
        "dp",
        "mns",
        "mnbt",
        "gmu",
        "bs",
        "kvc",
        "spec",
        "quant",
    ]
    # Full column order: config -> metrics -> totals -> path
    fieldnames = config_cols + [
        "gpus_for_1b_tokens_per_hour",
        "node_days_to_process_1b_tokens",
        "gpu_days_to_process_1b_tokens",
        "input_tps_per_gpu",
        "output_tps_per_gpu",
        "tokens_input_per_sec",
        "tokens_output_per_sec",
        "avg_running_reqs",
        "avg_waiting_reqs",
        "avg_gpu_kvc_usage",
        "avg_prefix_cache_hit_rate",
        "prompt_tokens_total",
        "completion_tokens_total",
        "mean_input_tokens",
        "mean_output_tokens",
        "successful_requests",
        "e2e_time",
        "inference_time",
        "e2e_per_1k_per_gpu",
        "inference_per_1k_per_gpu",
        "path",
        "comment",
    ]

    df = pd.DataFrame(rows, columns=fieldnames)
    # Ensure integer columns are properly typed
    for col in ["tp", "pp", "dp", "mns", "mnbt", "gmu", "bs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    metric_cols = [
        "node_days_to_process_1b_tokens",
        "gpu_days_to_process_1b_tokens",
        "input_tps_per_gpu",
        "output_tps_per_gpu",
        "e2e_time",
        "inference_time",
        "e2e_per_1k_per_gpu",
        "inference_per_1k_per_gpu",
    ]

    # Deduplicate: keep row with most metrics; if tied, keep highest output_tps_per_gpu; if still tied, keep most recent run
    def deduplicate(df_in: pd.DataFrame) -> pd.DataFrame:
        df_work = df_in.copy()
        df_work["__metric_count__"] = df_work[metric_cols].notna().sum(axis=1)
        df_work["__out_tok_tp__"] = df_work["output_tps_per_gpu"].fillna(-1)
        # Extract job ID from path (e.g., ".../slurm_logs/21899807_0.out" -> 21899807) for recency tiebreaker
        df_work["__job_id__"] = df_work["path"].str.extract(r"(\d+)_\d+\.out$").astype(float).fillna(0)
        return (
            df_work.sort_values(
                by=config_cols + ["__metric_count__", "__out_tok_tp__", "__job_id__"],
                ascending=[True] * len(config_cols) + [False, False, False],
            )
            .drop_duplicates(subset=config_cols, keep="first")
            .drop(columns=["__metric_count__", "__out_tok_tp__", "__job_id__"])
        )

    df_dedup = deduplicate(df)

    # Config columns for table display (may be constant or varying per experiment)
    config_columns = [
        ("prompt", "prompt_template_name", "left"),
        ("model", "model", "left"),
        ("tp", "tp", "right"),
        ("pp", "pp", "right"),
        ("dp", "dp", "right"),
        ("mns", "mns", "right"),
        ("mnbt", "mnbt", "right"),
        ("gmu", "gmu", "right"),
        ("bs", "bs", "right"),
        ("kvc", "kvc", "left"),
        ("spec", "spec", "left"),
        ("quant", "quant", "left"),
    ]
    # Metric columns always shown in table
    display_metric_columns = [
        ("gpus/1b/h", "gpus_for_1b_tokens_per_hour", "right"),
        ("out tps/gpu", "output_tps_per_gpu", "right"),
        ("mean_in", "mean_input_tokens", "right"),
        ("mean_out", "mean_output_tokens", "right"),
        ("kvc%", "avg_gpu_kvc_usage", "right"),
        ("prefix%", "avg_prefix_cache_hit_rate", "right"),
        ("inf/1k/gpu", "inference_per_1k_per_gpu", "right"),
        ("comment", "comment", "left"),
    ]

    def fmt(val: object, col_key: str) -> str:
        """Format a value for table display."""
        if val is None or pd.isna(val):
            return ""
        if isinstance(val, (int, float)):
            if col_key == "inference_per_1k_per_gpu":
                # Format as HH:MM:SS (GPU-time to process 1k examples)
                total_secs = int(float(val))
                hours, remainder = divmod(total_secs, 3600)
                minutes, seconds = divmod(remainder, 60)
                return f"{hours}:{minutes:02d}:{seconds:02d}"
            if col_key in ("node_days_to_process_1b_tokens", "gpu_days_to_process_1b_tokens"):
                return f"{float(val):.3f}"
            if col_key in ("avg_gpu_kvc_usage", "avg_prefix_cache_hit_rate"):
                return f"{float(val):.1f}"
            if col_key in (
                "input_tps_per_gpu",
                "output_tps_per_gpu",
                "mean_input_tokens",
                "mean_output_tokens",
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
        for prefix in [("spec", "spec_"), ("quant", "quant_"), ("kvc", "kvc_")]:
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

    def print_optimization_summary(df_all: pd.DataFrame) -> pd.DataFrame:
        """
        Print optimization summary comparing baseline vs best config for each model.

        Baseline defaults: tp=1, mns=256, mnbt=8192 (vLLM defaults).
        Aggregates across all experiments to find the best config per model.
        Returns a DataFrame with the summary for CSV export.
        """
        # Baseline defaults (vLLM defaults + common benchmark settings)
        baseline_defaults: dict[str, object] = {
            "tp": 1,
            "mns": 256,
            "mnbt": 8192,
            "gmu": 90,
            "bs": 16,
            "kvc": "kvc_auto",
            "spec": "spec_none",
            "quant": "quant_none",
        }

        summary_rows: list[dict[str, object]] = []

        for model in sorted(df_all["model"].unique()):
            df_model = df_all[df_all["model"] == model].copy()
            df_valid = df_model[df_model["output_tps_per_gpu"].notna()]

            if df_valid.empty:
                continue

            # Find baseline row matching all default parameters
            baseline_mask = pd.Series(True, index=df_valid.index)
            for param, default_val in baseline_defaults.items():
                baseline_mask &= df_valid[param] == default_val
            df_baseline = df_valid[baseline_mask]

            # Find best row (highest output_tps_per_gpu)
            best_idx = df_valid["output_tps_per_gpu"].idxmax()
            best_row = df_valid.loc[best_idx]
            best_tps = best_row["output_tps_per_gpu"]

            if df_baseline.empty:
                # No exact baseline match - use lowest config as proxy
                baseline_idx = df_valid["output_tps_per_gpu"].idxmin()
                baseline_row = df_valid.loc[baseline_idx]
            else:
                # Multiple baseline matches possible (different experiments) - pick best one
                baseline_idx = df_baseline["output_tps_per_gpu"].idxmax()
                baseline_row = df_baseline.loc[baseline_idx]
            baseline_tps = baseline_row["output_tps_per_gpu"]

            speedup = best_tps / baseline_tps if baseline_tps > 0 else None

            # Identify parameters that deviate from baseline defaults
            deviations: list[str] = []
            for param, default_val in baseline_defaults.items():
                best_val = best_row[param]
                if pd.notna(best_val) and best_val != default_val:
                    deviations.append(f"{param}={best_val}")

            summary_rows.append(
                {
                    "model": model,
                    "baseline_tps": int(round(baseline_tps)),
                    "optimized_tps": int(round(best_tps)),
                    "speedup": speedup,
                    "optimized_params": ", ".join(deviations) if deviations else "(baseline is optimal)",
                }
            )

        if not summary_rows:
            return pd.DataFrame()

        df_summary = pd.DataFrame(summary_rows)

        # Print summary table
        print(f"\n{'=' * 80}")
        print("Optimization Summary (All Models)")
        print(f"{'=' * 80}")
        baseline_str = ", ".join(f"{k}={v}" for k, v in baseline_defaults.items())
        print(f"Baseline: {baseline_str}")
        print()

        # Calculate column widths
        headers = ["Model", "Baseline tps/gpu", "Optimized tps/gpu", "Speedup", "Optimized Parameters"]
        col_keys = ["model", "baseline_tps", "optimized_tps", "speedup", "optimized_params"]

        def fmt_cell(val: object, key: str) -> str:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "-"
            if key == "speedup":
                return f"{val:.2f}x"
            return str(val)

        data_rows = [[fmt_cell(row[k], k) for k in col_keys] for row in summary_rows]
        widths = [max(len(h), max((len(r[i]) for r in data_rows), default=0)) for i, h in enumerate(headers)]

        # Print header
        print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
        print("| " + " | ".join("-" * w for w in widths) + " |")

        # Print data rows
        aligns = ["left", "right", "right", "right", "left"]
        for r in data_rows:
            cells = []
            for i, cell in enumerate(r):
                cells.append(cell.rjust(widths[i]) if aligns[i] == "right" else cell.ljust(widths[i]))
            print("| " + " | ".join(cells) + " |")

        return df_summary

    # Process each experiment
    total_rows = 0

    for experiment_name in sorted(df_dedup["experiment"].unique()):
        df_exp = df_dedup[df_dedup["experiment"] == experiment_name]
        print_table(df_exp, experiment_name)

        # Save CSV for this experiment
        exp_csv_path = root_path / f"{experiment_name}.csv"
        df_exp.to_csv(exp_csv_path, index=False, float_format="%.6f")
        logger.info(f"Wrote {len(df_exp)} rows to {exp_csv_path}")
        total_rows += len(df_exp)

    # Save combined CSV
    out_csv_path = root_path / "benchmarking_results.csv"
    df_dedup.to_csv(out_csv_path, index=False, float_format="%.6f")
    logger.info(
        f"Wrote {total_rows} total rows across {len(df_dedup['experiment'].unique())} experiments to {out_csv_path}"
    )

    # Print combined optimization summary across all models
    df_opt_summary = print_optimization_summary(df_dedup)
    if not df_opt_summary.empty:
        opt_csv_path = root_path / "optimization_summary.csv"
        df_opt_summary.to_csv(opt_csv_path, index=False)
        logger.info(f"Wrote optimization summary to {opt_csv_path}")

    return total_rows


def main(
    root: str = "examples/inference/benchmark/results",
    n_jobs: int = -1,
) -> None:
    """Analyze benchmark results.

    Args:
        root: Root directory containing experiment results.
        n_jobs: Number of parallel jobs for processing logs (-1 for all CPUs).
    """
    analyze(root, n_jobs)


if __name__ == "__main__":
    typer.run(main)
