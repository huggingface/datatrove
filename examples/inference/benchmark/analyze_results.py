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
from pathlib import Path

import pandas as pd
import typer

from datatrove.utils.logging import logger


GPUS_PER_NODE = 8

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
    content = server_log_path.read_text()
    return any(pattern.search(content) for pattern in OOM_PATTERNS)


def parse_server_logs(server_log_path: Path) -> dict[str, float] | None:
    """
    Parse vLLM server logs to extract throughput metrics, excluding server startup time.

    Returns:
        Dict with:
          - avg_prompt_throughput: Average prompt throughput from lines with prompt>0 (total for engine)
          - avg_generation_throughput: Average generation throughput from steady-state lines (total for engine)
        Or None if parsing fails

    Note: The returned values are the total throughput for the entire engine.
          Divide by TP to get per-GPU throughput.
    """
    try:
        if not server_log_path.exists():
            return None

        with open(server_log_path, "r") as f:
            log_lines = f.readlines()

        # Pattern to extract throughput metrics from vLLM logs
        # Example: "Avg prompt throughput: 6949.8 tokens/s, Avg generation throughput: 5108.0 tokens/s"
        pattern = re.compile(
            r"Avg prompt throughput:\s+([\d.]+)\s+tokens/s,\s+Avg generation throughput:\s+([\d.]+)\s+tokens/s"
        )

        prompt_throughputs = []
        generation_throughputs_steady = []
        generation_throughputs_mixed = []

        for line in log_lines:
            match = pattern.search(line)
            if match:
                prompt_thr = float(match.group(1))
                gen_thr = float(match.group(2))

                if prompt_thr == 0.0:
                    # Steady-state: pure generation phase
                    generation_throughputs_steady.append(gen_thr)
                else:
                    # Mixed phase: prompt processing happening
                    prompt_throughputs.append(prompt_thr)
                    generation_throughputs_mixed.append(gen_thr)

        # Calculate averages
        avg_prompt = sum(prompt_throughputs) / len(prompt_throughputs) if prompt_throughputs else None
        avg_gen_steady = (
            sum(generation_throughputs_steady) / len(generation_throughputs_steady)
            if generation_throughputs_steady
            else None
        )
        # Use mixed-phase generation throughput as fallback if no steady-state data
        avg_gen_mixed = (
            sum(generation_throughputs_mixed) / len(generation_throughputs_mixed)
            if generation_throughputs_mixed
            else None
        )

        if avg_prompt is None and avg_gen_steady is None and avg_gen_mixed is None:
            return None

        return {
            "avg_prompt_throughput": avg_prompt,
            # Prefer steady-state, fall back to mixed-phase for continuous workloads
            "avg_generation_throughput": avg_gen_steady if avg_gen_steady is not None else avg_gen_mixed,
        }
    except Exception:
        return None


def parse_path_fields(
    file_path: str,
    root: str = "examples/inference/benchmark/results",
) -> tuple[str, str, int | None, int | None, int | None, int | None, int | None, str, str, str]:
    """
    Parse experiment, model, tp, pp, dp, mns, mnbt, spec_config, quant_config, and kv_cache_config from the file path.

    Expected path structure:
        {root}/{experiment}/model_name/tp{X}-pp{Y}-dp{Z}/mns_{N}/mnbt_{M}/spec_{...}/quant_{...}/kv_{...}/inference_logs/...

    Returns:
        Tuple of (experiment, model, tp, pp, dp, mns, mnbt, spec_config, quant_config, kv_cache_config)
    """
    p = Path(file_path)
    parts = list(p.parts)
    experiment: str = ""
    model: str = ""
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    mns: int | None = None
    mnbt: int | None = None
    spec_config: str = ""
    quant_config: str = ""
    kv_cache_config: str = ""

    # Extract experiment name: first directory after root
    root_path = Path(root)
    root_parts = list(root_path.parts)
    if len(parts) > len(root_parts):
        experiment = parts[len(root_parts)]

    # Find a 'tp' segment anywhere in the path, supporting 'tp_8', 'tp-8', or 'tp8-pp2-dp1'
    tp_idx = None
    for i, part in enumerate(parts):
        # Match tp<digits> optionally followed by pp<digits> and dp<digits>
        m = re.match(r"^tp[_-]?(\d+)(?:[_-]?pp[_-]?(\d+))?(?:[_-]?dp[_-]?(\d+))?", part)
        if m:
            tp = int(m.group(1))
            if m.group(2):
                pp = int(m.group(2))
            if m.group(3):
                dp = int(m.group(3))
            tp_idx = i
            break

    # Heuristic: model is the directory right before the 'tp_*' segment
    if tp_idx is not None and tp_idx - 1 >= 0:
        model = parts[tp_idx - 1]

    # Parse mns segment (position tp_idx + 1)
    if tp_idx is not None and tp_idx + 1 < len(parts):
        mns_part = parts[tp_idx + 1]
        if isinstance(mns_part, str) and mns_part.startswith("mns_"):
            try:
                mns = int(mns_part[4:])
            except ValueError:
                mns = None

    # Parse mnbt segment (position tp_idx + 2)
    if tp_idx is not None and tp_idx + 2 < len(parts):
        mnbt_part = parts[tp_idx + 2]
        if isinstance(mnbt_part, str) and mnbt_part.startswith("mnbt_"):
            try:
                mnbt = int(mnbt_part[5:])
            except ValueError:
                mnbt = None

    # Heuristic: spec segment comes at position tp_idx + 3
    if tp_idx is not None and tp_idx + 3 < len(parts):
        spec_part = parts[tp_idx + 3]
        if isinstance(spec_part, str) and spec_part.startswith("spec_"):
            spec_config = spec_part
        else:
            spec_config = "spec_none"

    # Heuristic: quant segment comes at position tp_idx + 4
    if tp_idx is not None and tp_idx + 4 < len(parts):
        quant_part = parts[tp_idx + 4]
        if isinstance(quant_part, str) and quant_part.startswith("quant_"):
            quant_config = quant_part
        else:
            quant_config = "quant_none"
    else:
        quant_config = "quant_none"

    # Heuristic: kv_cache segment comes at position tp_idx + 5
    if tp_idx is not None and tp_idx + 5 < len(parts):
        kv_part = parts[tp_idx + 5]
        if isinstance(kv_part, str) and kv_part.startswith("kv_"):
            kv_cache_config = kv_part
        else:
            kv_cache_config = "kv_auto"
    else:
        kv_cache_config = "kv_auto"

    return experiment, model, tp, pp, dp, mns, mnbt, spec_config, quant_config, kv_cache_config


def _get_total_from_stat_field(val: object) -> float | None:
    """
    Normalize a stats.json field that may be a number or a dict with a 'total' key.
    Returns a float or None.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        t = val.get("total")
        if isinstance(t, (int, float)):
            return float(t)
    return None


def parse_stats_json(stats_path: Path) -> dict[str, float] | None:
    """
    Parse a merged stats.json produced by datatrove.tools.merge_stats to extract:
      - input_tokens_total (prompt_tokens.total)
      - output_tokens_total (completion_tokens.total)
      - successful_requests_total
      - failed_requests_total (if present; else 0)

    Note: Does NOT extract timing information - use server logs for throughput metrics.
    """
    try:
        if not stats_path.exists():
            return None
        with open(stats_path, "r") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            return None
        # Find the "Model call: Inference" entry
        entry: dict | None = None
        for item in data:
            if isinstance(item, dict) and "name" in item:
                name = str(item.get("name", ""))
                if "Model call" in name:
                    entry = item
                    break
        if not entry:
            return None
        st = entry.get("stats") or {}
        in_total = _get_total_from_stat_field(st.get("prompt_tokens"))
        out_total = _get_total_from_stat_field(st.get("completion_tokens"))
        succ = _get_total_from_stat_field(st.get("successful_requests"))
        fail_val = st.get("failed_requests", 0)
        fail = _get_total_from_stat_field(fail_val)
        if fail is None:
            fail = float(fail_val) if isinstance(fail_val, (int, float)) else 0.0
        return {
            "input_tokens_total": in_total or 0.0,
            "output_tokens_total": out_total or 0.0,
            "successful_requests_total": int(succ) if isinstance(succ, (int, float)) else 0,
            "failed_requests_total": int(fail) if isinstance(fail, (int, float)) else 0,
        }
    except Exception:
        return None


def compute_days_for_1b_from_per_gpu(output_tps_per_gpu: float | None) -> tuple[float | None, float | None]:
    """
    Compute GPU-days and node-days to process 1e9 output tokens using per-TP (per-GPU) throughput.
    - output_tps_per_gpu: per-TP output tokens per second (lifetime)
      GPU-days is the time in days for a single GPU at this rate to generate 1e9 tokens.
      Node-days scales by 8 GPUs per node.
    """
    if not output_tps_per_gpu or output_tps_per_gpu <= 0:
        return None, None
    seconds = 1_000_000_000.0 / output_tps_per_gpu
    days = seconds / 86400.0
    gpu_days = days
    node_days = days / GPUS_PER_NODE
    return gpu_days, node_days


def compute_gpus_for_1b_per_hour_from_per_gpu(output_tps_per_gpu: float | None) -> float | None:
    """
    Compute the number of GPUs required to generate 1e9 output tokens in one hour,
    given the per-TP (per-GPU) output tokens per second.
    """
    if not output_tps_per_gpu or output_tps_per_gpu <= 0:
        return None
    per_gpu_tokens_per_hour = output_tps_per_gpu * 3600.0
    return 1_000_000_000.0 / per_gpu_tokens_per_hour


def analyze(root: str, out_csv: str) -> int:
    files = sorted(glob.glob(os.path.join(root, "**", "inference_logs", "slurm_logs", "*.out"), recursive=True))
    rows: list[dict[str, object]] = []

    for path in files:
        experiment, model, tp, pp, dp, mns, mnbt, spec, quant, kv = parse_path_fields(path, root)

        # Create row template with default None values
        # Note: util is kept in CSV but not parsed from path
        row = {
            "experiment": experiment,
            "model": model,
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "mns": mns,
            "mnbt": mnbt,
            "util": None,
            "spec": spec,
            "quant": quant,
            "kv": kv,
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
            sorted(server_logs_dir.glob("server_rank_*_node_*.log"))
            or sorted(server_logs_dir.glob("server_rank_*.log"))
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

        row["prompt_tokens_total"] = int(stats_info.get("input_tokens_total") or 0)
        row["completion_tokens_total"] = int(stats_info.get("output_tokens_total") or 0)
        row["successful_requests"] = stats_info.get("successful_requests_total")
        row["failed_requests"] = stats_info.get("failed_requests_total")

        server_metrics = parse_server_logs(server_log_path) if server_log_path else None

        if server_metrics is None:
            row["comment"] = "server logs not found or unparseable"
            rows.append(row)
            continue

        # Server log metrics are total throughput for the engine
        # Divide by total GPUs (TP * PP) to get per-GPU throughput
        avg_prompt_thr = server_metrics.get("avg_prompt_throughput")
        avg_gen_thr = server_metrics.get("avg_generation_throughput")

        if avg_prompt_thr is None or avg_gen_thr is None:
            row["comment"] = "throughput metrics not found in server logs"
            rows.append(row)
            continue

        # Normalize tp/pp to 1 if None or 0
        tp_val = tp if tp and tp > 0 else 1
        pp_val = pp if pp and pp > 0 else 1
        # dp does not affect per-instance throughput if we are looking at one instance log

        # Total GPUs contributing to this throughput = TP * PP
        gpu_divisor = tp_val * pp_val

        row["tokens_input_per_sec"] = avg_prompt_thr
        row["input_tps_per_gpu"] = avg_prompt_thr / gpu_divisor

        row["tokens_output_per_sec"] = avg_gen_thr
        row["output_tps_per_gpu"] = avg_gen_thr / gpu_divisor

        gpu_days, node_days = compute_days_for_1b_from_per_gpu(row["output_tps_per_gpu"])
        row["gpu_days_to_process_1b_tokens"] = gpu_days
        row["node_days_to_process_1b_tokens"] = node_days
        row["gpus_for_1b_tokens_per_hour"] = compute_gpus_for_1b_per_hour_from_per_gpu(row["output_tps_per_gpu"])

        # Build comment (warn on failed requests)
        if isinstance(row["failed_requests"], int) and row["failed_requests"] > 0:
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

    # Build pandas DataFrame
    df = pd.DataFrame(rows, columns=fieldnames)
    # Ensure integer columns are properly typed (no floats like 1024.0)
    for col in ["tp", "pp", "dp", "mns", "mnbt"]:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        except Exception:
            pass
    # For CSV: ensure one row per actual experiment folder (model/tp/spec_config).
    metric_cols_csv = [
        "node_days_to_process_1b_tokens",
        "gpu_days_to_process_1b_tokens",
        "input_tps_per_gpu",
        "output_tps_per_gpu",
    ]
    # Columns used for deduplication (unique experiment identifier)
    dedup_cols = ["experiment", "model", "tp", "pp", "dp", "mns", "mnbt", "spec", "quant", "kv"]

    df_out = df.copy()
    df_out["__metric_count__"] = df_out[metric_cols_csv].notna().sum(axis=1)
    df_out["__out_tok_tp__"] = df_out["output_tps_per_gpu"].fillna(-1)
    df_out = (
        df_out.sort_values(
            by=dedup_cols + ["__metric_count__", "__out_tok_tp__"],
            ascending=[True] * len(dedup_cols) + [False, False],
        )
        .drop_duplicates(subset=dedup_cols, keep="first")
        .drop(columns=["__metric_count__", "__out_tok_tp__"], errors="ignore")
    )

    # Column mapping for table display (excludes experiment - shown in header)
    # Config columns that may be constant or varying per experiment
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
    metric_columns = [
        ("gpus/1b/h", "gpus_for_1b_tokens_per_hour", "right"),
        ("in tps/gpu", "input_tps_per_gpu", "right"),
        ("out tps/gpu", "output_tps_per_gpu", "right"),
        ("comment", "comment", "left"),
    ]

    # Prepare display data:
    # - keep only rows that have at least one of the displayed metrics present
    # - for duplicate experiment combinations, keep the row with the most
    #   available displayed metrics; if tied, keep the one with the highest out-tok/tp
    metric_cols = [
        "node_days_to_process_1b_tokens",
        "gpu_days_to_process_1b_tokens",
        "input_tps_per_gpu",
        "output_tps_per_gpu",
    ]
    df_disp = df.copy()
    df_disp["__metric_count__"] = df_disp[metric_cols].notna().sum(axis=1)
    df_disp["__out_tok_tp__"] = df_disp["output_tps_per_gpu"].fillna(-1)
    df_disp = df_disp.sort_values(
        by=dedup_cols + ["__metric_count__", "__out_tok_tp__"],
        ascending=[True] * len(dedup_cols) + [False, False],
    ).drop_duplicates(subset=dedup_cols, keep="first")
    # Do not drop rows with no metrics; we still want one row per existing experiment folder

    def fmt(val: object, col_key: str) -> str:
        if val is None:
            return ""
        try:
            import pandas as _pd  # avoid shadowing

            if _pd.isna(val):
                return ""
        except Exception:
            pass
        # Column-aware numeric formatting for the summary table
        try:
            from numbers import Number

            is_num = isinstance(val, Number)
        except Exception:
            is_num = False
        if is_num:
            if col_key in ("node_days_to_process_1b_tokens", "gpu_days_to_process_1b_tokens"):
                # Show 3 decimals for day metrics
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
                # Show integer (rounded) for token rates per TP and batch size params
                try:
                    return f"{int(round(float(val)))}"
                except Exception:
                    return ""
            # Fallback: one decimal for any other numeric column
            return f"{float(val):.1f}"
        # Strip prefixes from config columns for narrower display
        str_val = str(val)
        if col_key == "spec" and str_val.startswith("spec_"):
            return str_val[5:]  # Remove "spec_" prefix
        if col_key == "quant" and str_val.startswith("quant_"):
            return str_val[6:]  # Remove "quant_" prefix
        if col_key == "kv" and str_val.startswith("kv_"):
            return str_val[3:]  # Remove "kv_" prefix
        return str_val

    def print_table(df_exp: pd.DataFrame, experiment_name: str) -> None:
        """Print a formatted table for a single experiment with dynamic columns."""
        # Identify which config columns have varying vs constant values
        varying_configs: list[tuple[str, str, str]] = []
        constant_configs: list[tuple[str, str]] = []  # (label, value)

        for label, col_key, align in config_columns:
            unique_vals = df_exp[col_key].dropna().unique()
            if len(unique_vals) <= 1:
                # Constant column - show in header
                val = fmt(unique_vals[0], col_key) if len(unique_vals) == 1 else ""
                if val:  # Only include if there's a value
                    constant_configs.append((label, val))
            else:
                # Varying column - show in table
                varying_configs.append((label, col_key, align))

        # Build dynamic columns_map for this experiment
        dynamic_columns = varying_configs + metric_columns

        # Collect rows for width calc
        header_labels = [h for (h, _, _) in dynamic_columns]
        data_rows: list[list[str]] = []
        for _, row in df_exp.iterrows():
            rendered = [fmt(row[col_key], col_key) for _, col_key, _ in dynamic_columns]
            data_rows.append(rendered)

        # Compute widths
        widths = [len(h) for h in header_labels]
        for r in data_rows:
            for i, cell in enumerate(r):
                if len(cell) > widths[i]:
                    widths[i] = len(cell)

        # Printer with alignment
        def pad(cell: str, width: int, align: str) -> str:
            return cell.rjust(width) if align == "right" else cell.ljust(width)

        aligns = [align for (_, _, align) in dynamic_columns]

        # Print experiment header with constant config values
        print(f"\n{'=' * 60}")
        header_parts = [f"Experiment: {experiment_name}"]
        if constant_configs:
            constants_str = ", ".join(f"{label}={val}" for label, val in constant_configs)
            header_parts.append(f"[{constants_str}]")
        print(" ".join(header_parts))
        print(f"{'=' * 60}")

        # Header
        header_line = (
            "| "
            + " | ".join(
                pad(h, widths[i], "left" if aligns[i] == "left" else "right") for i, h in enumerate(header_labels)
            )
            + " |"
        )
        sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
        print(header_line)
        print(sep_line)
        # Rows
        for r in data_rows:
            line = "| " + " | ".join(pad(r[i], widths[i], aligns[i]) for i in range(len(r))) + " |"
            print(line)

    # Get unique experiments and process each one
    experiments = df_disp["experiment"].unique()
    out_csv_dir = Path(out_csv).parent
    total_rows = 0

    for experiment_name in sorted(experiments):
        # Filter data for this experiment
        df_exp_disp = df_disp[df_disp["experiment"] == experiment_name]
        df_exp_out = df_out[df_out["experiment"] == experiment_name]

        # Print table for this experiment
        print_table(df_exp_disp, experiment_name)

        # Save CSV for this experiment (named by experiment)
        exp_csv_path = out_csv_dir / f"{experiment_name}.csv"
        df_exp_out.to_csv(exp_csv_path, index=False, float_format="%.6f")
        logger.info(f"Wrote {len(df_exp_out)} rows to {exp_csv_path}")
        total_rows += len(df_exp_out)

    # Also save combined CSV with all experiments
    df_out.to_csv(out_csv, index=False, float_format="%.6f")
    logger.info(f"Wrote {total_rows} total rows across {len(experiments)} experiments to {out_csv}")

    return total_rows


def main(
    root: str = "examples/inference/benchmark/results",
    out_csv: str = "examples/inference/benchmark/results/benchmarking_results.csv",
) -> None:
    analyze(root, out_csv)


if __name__ == "__main__":
    typer.run(main)
