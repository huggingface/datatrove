#!/usr/bin/env python3
"""
Launch benchmark experiments from a YAML configuration file.

A dry run prints the Slurm submission commands and planned job submissions
without actually submitting any jobs or making changes.

Usage:

python examples/inference/benchmark/launch_experiments.py \
    --config examples/inference/benchmark/sample_benchmark_config.yaml

python examples/inference/benchmark/launch_experiments.py \
    --config examples/inference/benchmark/sample_benchmark_config.yaml --dry-run

Retry only OOM failures (skip timeout and server_fail but re-run OOM):

python examples/inference/benchmark/launch_experiments.py \
    --config examples/inference/benchmark/sample_benchmark_config.yaml \
    --skip-failure-reasons timeout,server_fail

Re-run all previously failed experiments (skip nothing):

python examples/inference/benchmark/launch_experiments.py \
    --config examples/inference/benchmark/sample_benchmark_config.yaml \
    --skip-failure-reasons none
"""

import re
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import typer
import yaml

from datatrove.utils.logging import logger


sys.path.insert(0, str(Path(__file__).parent.parent))
# Import generate_data.main directly to avoid subprocess overhead (~10s per invocation)
from generate_data import main as generate_data_main
from utils import (
    build_run_path,
    detect_failure_reason,
    encode_bs_segment_for_log_dir,
    encode_gmu_segment_for_log_dir,
    encode_kvc_segment_for_log_dir,
    encode_mnbt_segment_for_log_dir,
    encode_mns_segment_for_log_dir,
    encode_quant_segment_for_log_dir,
    encode_spec_segment_for_log_dir,
    model_name_safe,
    normalize_kvc_dtype,
    normalize_quantization,
    normalize_speculative,
)


class ExperimentLauncher:
    """Launches experiments based on YAML configuration."""

    # All recognized failure reasons from detect_failure_reason
    ALL_FAILURE_REASONS = {"OOM", "timeout", "server_fail"}

    def __init__(
        self,
        config_path: str,
        dry_run: bool = False,
        skip_failure_reasons: set[str] | None = None,
    ):
        """
        Initialize the experiment launcher.

        Args:
            config_path: Path to YAML configuration file
            dry_run: If True, print commands without executing
            skip_failure_reasons: Failure reasons that cause a run to be skipped.
                Defaults to all reasons: {"OOM", "timeout", "server_fail"}.
        """
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.skip_failure_reasons = (
            skip_failure_reasons if skip_failure_reasons is not None else self.ALL_FAILURE_REASONS
        )

        # Load and validate configuration
        self.config = self._load_config()
        self._validate_config()

        # Setup experiment metadata
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def _load_config(self) -> dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Empty configuration file")

        return config

    def _validate_config(self) -> None:
        """Validate the configuration structure."""
        required_keys = ["script", "experiments"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in config: {key}")

        # Validate experiments structure
        if not isinstance(self.config["experiments"], list):
            raise ValueError("'experiments' must be a list of dictionaries")

        if len(self.config["experiments"]) == 0:
            raise ValueError("'experiments' list cannot be empty")

        for i, experiment in enumerate(self.config["experiments"]):
            if not isinstance(experiment, dict):
                raise ValueError(f"Experiment {i} must be a dictionary")
            if "name" not in experiment:
                raise ValueError(f"Experiment {i} missing required 'name' field")
            # Validate that no experiment has 'name' in its args (since we auto-generate run names)
            if "name" in (experiment.get("args") or {}):
                raise ValueError(
                    f"Experiment '{experiment['name']}' should not have 'name' in its args. "
                    f"Run names are automatically derived from the configuration."
                )

    def _sanitize_for_name(self, value: Any) -> str:
        """Sanitize a value to be safely embedded into a run name."""
        if isinstance(value, bool):
            val_str = "true" if value else "false"
        elif isinstance(value, float):
            val_str = f"{value:.6g}"
        else:
            val_str = str(value)

        # Replace path separators to avoid directory-like names
        val_str = val_str.replace("/", "-")
        # Keep only safe characters
        val_str = re.sub(r"[^A-Za-z0-9._-]+", "_", val_str)
        # Collapse multiple underscores and trim
        val_str = re.sub(r"_{2,}", "_", val_str).strip("_")
        return val_str

    def _derive_run_name(self, args: dict[str, Any]) -> str:
        """Derive run name from non-default config values, matching path order.

        Order: prompt / model / tp-pp-dp / mns / mnbt / gmu / bs / kvc / spec / quant
        Only includes segments that differ from defaults to keep names short.
        """
        parts: list[str] = []

        # 1. Prompt template name (if specified as [name, template] tuple)
        prompt_template = args.get("prompt-template") or args.get("prompt_template")
        if isinstance(prompt_template, list) and len(prompt_template) >= 1:
            parts.append(self._sanitize_for_name(prompt_template[0]))

        # 2. Model name (always included)
        model_value = args.get("model-name-or-path") or args.get("model_name_or_path") or "Qwen/Qwen3-0.6B"
        parts.append(model_name_safe(str(model_value)))

        # 3. Parallelism settings (defaults: tp=1, pp=1, dp=1)
        tp, pp, dp = args.get("tp") or 1, args.get("pp") or 1, args.get("dp") or 1
        if tp != 1:
            parts.append(f"tp{tp}")
        if pp != 1:
            parts.append(f"pp{pp}")
        if dp != 1:
            parts.append(f"dp{dp}")

        # 4. Batch size parameters (defaults: mns=256, mnbt=8192)
        mns = int(args.get("max-num-seqs") or args.get("max_num_seqs") or 256)
        mnbt = int(args.get("max-num-batched-tokens") or args.get("max_num_batched_tokens") or 8192)
        if mns != 256:
            parts.append(encode_mns_segment_for_log_dir(mns))
        if mnbt != 8192:
            parts.append(encode_mnbt_segment_for_log_dir(mnbt))

        # 5. GPU memory utilization (default: 0.9)
        gmu = float(args.get("gpu-memory-utilization") or args.get("gpu_memory_utilization") or 0.9)
        if gmu != 0.9:
            parts.append(encode_gmu_segment_for_log_dir(gmu))

        # 6. Block size (default: 16)
        bs = int(args.get("block-size") or args.get("block_size") or 16)
        if bs != 16:
            parts.append(encode_bs_segment_for_log_dir(bs))

        # 7. KV cache dtype config (default: "auto")
        kv_raw = args.get("kv-cache-dtype") or args.get("kv_cache_dtype") or "auto"
        if kv_raw.strip().lower() not in ("auto", "none", "null", ""):
            kv_norm = normalize_kvc_dtype(kv_raw)
            parts.append(encode_kvc_segment_for_log_dir(kv_norm))

        # 8. Speculative config (default: None)
        spec_raw = args.get("speculative-config") or args.get("speculative_config")
        if isinstance(spec_raw, str) and spec_raw.strip().lower() in ("none", "null", ""):
            spec_raw = None
        if spec_raw:
            spec_norm = normalize_speculative(spec_raw)
            parts.append(encode_spec_segment_for_log_dir(spec_norm))

        # 9. Quantization config (default: None)
        quant_raw = args.get("quantization")
        if isinstance(quant_raw, str) and quant_raw.strip().lower() in ("none", "null", ""):
            quant_raw = None
        if quant_raw:
            quant_norm = normalize_quantization(quant_raw)
            parts.append(encode_quant_segment_for_log_dir(quant_norm))

        return "-".join(parts)

    @staticmethod
    def _is_sweep_value(key: str, value: Any) -> bool:
        """Determine if a value should be treated as a sweep parameter.

        prompt-template can be either:
          - A single template: ["name", "template string"] -> NOT a sweep
          - Multiple templates: [["math", "..."], ["table", "..."]] -> IS a sweep
        All other list values are always sweeps.
        """
        if not isinstance(value, list) or len(value) == 0:
            return False
        if key in ("prompt-template", "prompt_template"):
            # Single [name, template] pair: list of strings -> not a sweep
            # Multiple templates: list of lists -> sweep
            return isinstance(value[0], list)
        return True

    def _expand_experiment(self, experiment_config: dict[str, Any]) -> list[dict[str, Any]]:
        """Expand a single experiment config into multiple runs if any args values are lists.

        If multiple args contain lists (in either fixed_args or experiment args),
        generate the cartesian product (all permutations). Each run corresponds to one SLURM job.
        Experiment args override fixed_args for the same key.

        prompt-template is special: a single [name, template] pair is not a sweep,
        but a list of such pairs (list of lists) is expanded as a sweep.
        """
        fixed_args = self.config.get("fixed_args") or {}
        exp_args = experiment_config.get("args") or {}
        experiment_name = experiment_config["name"]

        # Merge fixed_args with exp_args (exp_args take precedence)
        merged_args = {**fixed_args, **exp_args}

        # Find all sweep keys
        sweep_keys = [key for key, value in merged_args.items() if self._is_sweep_value(key, value)]

        if not sweep_keys:
            derived_name = self._derive_run_name(merged_args)
            return [
                {
                    "name": derived_name,
                    "experiment": experiment_name,
                    "args": merged_args,
                }
            ]

        ordered_keys = sorted(sweep_keys)
        value_lists = [merged_args[key] for key in ordered_keys]

        expanded_runs: list[dict[str, Any]] = []
        for combo in product(*value_lists):
            new_args = dict(merged_args)
            for key, val in zip(ordered_keys, combo):
                new_args[key] = val

            new_name = self._derive_run_name(new_args)

            expanded_runs.append(
                {
                    "name": new_name,
                    "experiment": experiment_name,
                    "args": new_args,
                }
            )

        return expanded_runs

    def _build_kwargs(self, run_config: dict[str, Any]) -> dict[str, Any]:
        """Build kwargs dict for calling generate_data.main directly."""
        # Args are already merged in _expand_experiment
        merged = {**(run_config.get("args") or {}), "name": run_config["name"]}

        # Prepend experiment name to output-dir for directory grouping
        if "experiment" in run_config and "output-dir" in merged:
            merged["output-dir"] = f"{merged['output-dir']}/{run_config['experiment']}"

        # Convert CLI-style keys (with dashes) to Python kwargs (with underscores)
        return {key.replace("-", "_"): value for key, value in merged.items()}

    def _get_run_path(self, kwargs: dict[str, Any]) -> Path:
        """Build the run path from kwargs."""
        prompt_template = kwargs.get("prompt_template")
        prompt_name = prompt_template[0] if isinstance(prompt_template, list) else "default"

        return build_run_path(
            output_dir=kwargs.get("output_dir", ""),
            prompt_template_name=prompt_name,
            model_name_or_path=kwargs.get("model_name_or_path", ""),
            tp=kwargs.get("tp") or 1,
            pp=kwargs.get("pp") or 1,
            dp=kwargs.get("dp") or 1,
            max_num_seqs=int(kwargs.get("max_num_seqs") or 256),
            max_num_batched_tokens=int(kwargs.get("max_num_batched_tokens") or 8192),
            gpu_memory_utilization=float(kwargs.get("gpu_memory_utilization") or 0.9),
            block_size=int(kwargs.get("block_size") or 16),
            kv_cache_dtype=kwargs.get("kv_cache_dtype") or "auto",
            speculative_config=kwargs.get("speculative_config"),
            quantization=kwargs.get("quantization"),
        )

    def _is_already_completed(self, kwargs: dict[str, Any]) -> bool:
        """Check if a run already has a stats.json file (completed successfully)."""
        run_path = self._get_run_path(kwargs)
        return (run_path / "inference_logs" / "stats.json").exists()

    def _get_previous_failure_reason(self, kwargs: dict[str, Any]) -> str | None:
        """Check if a previous run failed with a skipable reason by scanning log files."""
        if not self.skip_failure_reasons:
            return None

        inference_logs = self._get_run_path(kwargs) / "inference_logs"
        if not inference_logs.exists():
            return None

        log_files = list(inference_logs.glob("slurm_logs/*.out")) + list(inference_logs.glob("server_logs/*.log"))
        for f in log_files:
            reason = detect_failure_reason(f)
            if reason in self.skip_failure_reasons:
                return reason
        return None

    def _execute_direct(self, kwargs: dict[str, Any], run_name: str) -> tuple[int, bool, str | None]:
        """Execute generate_data.main directly (no subprocess overhead).

        Returns:
            Tuple of (exit_code, skipped, job_id)
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Submitting Slurm job for run: {run_name}")
        logger.info(f"{'=' * 60}")

        if self.dry_run:
            logger.info("[DRY RUN] Slurm job would be submitted")
            return 0, False, None

        try:
            job_id = generate_data_main(**kwargs)
            return 0, False, job_id
        except KeyboardInterrupt:
            logger.info(f"\nInterrupted during job submission for run: {run_name}")
            raise
        except Exception as e:
            error_msg = str(e)
            if "Skipping launch" in error_msg or "already been completed" in error_msg:
                logger.info(f"Job skipped: {error_msg}")
                return 0, True, None
            logger.error(f"Error submitting job for run {run_name}: {e}")
            return 1, False, None

    def launch(self) -> dict[str, int]:
        """
        Launch all configured experiments as Slurm jobs.

        Returns:
            Dictionary mapping run names to job submission exit codes
        """
        logger.info(f"Configuration: {self.config_path}")
        logger.info(f"Timestamp: {self.timestamp}")
        if self.skip_failure_reasons:
            logger.info(f"Skipping previous failures: {sorted(self.skip_failure_reasons)}")
        else:
            logger.info("Not skipping any previous failures (will re-run all)")

        if self.dry_run:
            logger.info("\n*** DRY RUN MODE - No Slurm jobs will be submitted ***")
        else:
            logger.info("\n*** SUBMITTING SLURM JOBS ***")

        results = {}
        skip_reasons: dict[str, str] = {}  # run_name -> reason for skipping
        job_ids: list[str] = []

        # Expand experiments into runs (lists in args produce cartesian product of values)
        expanded_runs: list[dict[str, Any]] = []
        for experiment in self.config["experiments"]:
            expanded_runs.extend(self._expand_experiment(experiment))

        for run_config in expanded_runs:
            run_name = run_config["name"]

            # Build kwargs and call generate_data.main directly (no subprocess overhead)
            kwargs = self._build_kwargs(run_config)

            # Skip runs that already have stats.json (completed successfully)
            if self._is_already_completed(kwargs):
                results[run_name] = 0
                skip_reasons[run_name] = "completed"
                logger.info(f"⏭️ Skipping {run_name} (stats.json exists)")
                continue

            # Skip runs that previously failed with a non-recoverable reason
            failure_reason = self._get_previous_failure_reason(kwargs)
            if failure_reason:
                results[run_name] = 0
                skip_reasons[run_name] = failure_reason
                logger.info(f"⏭️ Skipping {run_name} (previous {failure_reason} failure)")
                continue

            exit_code, skipped, job_id = self._execute_direct(kwargs, run_name)
            results[run_name] = exit_code
            if skipped:
                skip_reasons[run_name] = "already completed"
            if job_id:
                job_ids.append(str(job_id))

            if exit_code != 0:
                logger.error(f"❌ Slurm job submission failed for {run_name} (exit code {exit_code})")

                # Check if we should continue on failure
                if not self.config.get("continue_on_failure", True):
                    logger.warning("Stopping job submissions due to failure")
                    break
            elif not self.dry_run:
                if skipped:
                    logger.info(f"ℹ️ Slurm job skipped for {run_name} (already completed)")
                else:
                    logger.info(f"✅ Slurm job submitted successfully for {run_name}")

        # Print summary
        logger.info(f"\n{'=' * 60}")
        logger.info("JOB SUBMISSION SUMMARY")
        logger.info(f"{'=' * 60}")

        for run_name, exit_code in results.items():
            if run_name in skip_reasons:
                status = f"⏭️ SKIPPED ({skip_reasons[run_name]})"
            elif self.dry_run:
                status = "✅ WOULD SUBMIT" if exit_code == 0 else f"❌ WOULD FAIL ({exit_code})"
            else:
                status = "✅ SUBMITTED" if exit_code == 0 else f"❌ FAILED ({exit_code})"
            logger.info(f"{run_name:<30} {status}")

        if results:
            successful_submissions = sum(1 for name, code in results.items() if code == 0 and name not in skip_reasons)
            skipped_count = len(skip_reasons)
            if self.dry_run:
                logger.info(f"\n{successful_submissions}/{len(results)} jobs would be submitted")
                if skipped_count > 0:
                    logger.info(f"{skipped_count} job(s) skipped (completed or previous failure)")
            else:
                logger.info(f"\n{successful_submissions}/{len(results)} jobs submitted successfully")
                if skipped_count > 0:
                    logger.info(f"{skipped_count} job(s) skipped (completed or previous failure)")
                if successful_submissions > 0:
                    logger.info("Use 'squeue -u $USER' to monitor job status")
                    if job_ids:
                        logger.info(f"Cancel all launched jobs: scancel {' '.join(job_ids)}")

        return results


def main(
    config: str = "examples/inference/benchmark/sample_benchmark_config.yaml",
    dry_run: bool = False,
    skip_failure_reasons: str = "OOM,timeout,server_fail",
) -> None:
    # "none" sentinel means skip nothing (re-run all failures)
    reasons = (
        set()
        if skip_failure_reasons.strip().lower() == "none"
        else {r.strip() for r in skip_failure_reasons.split(",") if r.strip()}
    )
    try:
        launcher = ExperimentLauncher(
            config_path=config,
            dry_run=dry_run,
            skip_failure_reasons=reasons,
        )

        results = launcher.launch()

        # Exit with non-zero code if any job submissions failed
        failed_submissions = [name for name, code in results.items() if code != 0]
        if failed_submissions:
            if dry_run:
                logger.info(f"\nDry run completed with {len(failed_submissions)} jobs that would fail")
            else:
                logger.info(f"\nExperiment completed with {len(failed_submissions)} failed job submissions")
            raise typer.Exit(code=1)
        else:
            if dry_run:
                logger.info(f"\nDry run completed - all {len(results)} jobs would submit successfully")
            else:
                logger.info(f"\nAll {len(results)} Slurm jobs submitted successfully")
            return

    except KeyboardInterrupt:
        logger.info("\nExperiment interrupted by user")
        raise typer.Exit(code=130)
    except typer.Exit as e:
        # Re-raise Typer's own Exit without printing an error message
        raise e
    except Exception as e:
        logger.error(f"Error: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
