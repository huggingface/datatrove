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
    encode_kv_cache_segment_for_log_dir,
    encode_mnbt_segment_for_log_dir,
    encode_mns_segment_for_log_dir,
    encode_quant_segment_for_log_dir,
    encode_spec_segment_for_log_dir,
    model_name_safe,
    normalize_kv_cache_dtype,
    normalize_quantization,
    normalize_speculative,
)


class ExperimentLauncher:
    """Launches experiments based on YAML configuration."""

    def __init__(self, config_path: str, dry_run: bool = False):
        """
        Initialize the experiment launcher.

        Args:
            config_path: Path to YAML configuration file
            dry_run: If True, print commands without executing
        """
        self.config_path = Path(config_path)
        self.dry_run = dry_run

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
            if "name" in experiment.get("args", {}):
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

    def _derive_run_name(self, args: dict[str, Any], fallback: str) -> str:
        """Derive run name as {model}-tp{tp}-pp{pp}-dp{dp}-{mns}-{mnbt}-{spec}-{quant}-{kv}."""
        model_value = args.get("model-name-or-path") or args.get("model_name_or_path") or fallback
        model_for_name = model_name_safe(str(model_value))
        tp_value = args.get("tp")
        tp_str = str(tp_value) if tp_value is not None else "1"
        pp_value = args.get("pp")
        pp_str = str(pp_value) if pp_value is not None else "1"
        dp_value = args.get("dp")
        dp_str = str(dp_value) if dp_value is not None else "1"

        # Handle batch size parameters
        mns_raw = args.get("max-num-seqs") or args.get("max_num_seqs") or 1000
        mns_short = encode_mns_segment_for_log_dir(int(mns_raw))
        mnbt_raw = args.get("max-num-batched-tokens") or args.get("max_num_batched_tokens") or 8192
        mnbt_short = encode_mnbt_segment_for_log_dir(int(mnbt_raw))

        # Handle speculative config
        spec_raw = args.get("speculative-config") or args.get("speculative_config")
        if isinstance(spec_raw, str) and spec_raw.strip().lower() in ("none", "null", ""):
            spec_raw = None
        spec_norm = normalize_speculative(spec_raw)
        spec_short = encode_spec_segment_for_log_dir(spec_norm)

        # Handle quantization config
        quant_raw = args.get("quantization")
        if isinstance(quant_raw, str) and quant_raw.strip().lower() in ("none", "null", ""):
            quant_raw = None
        quant_norm = normalize_quantization(quant_raw)
        quant_short = encode_quant_segment_for_log_dir(quant_norm)

        # Handle KV cache dtype config
        kv_raw = args.get("kv-cache-dtype") or args.get("kv_cache_dtype")
        kv_norm = normalize_kv_cache_dtype(kv_raw)
        kv_short = encode_kv_cache_segment_for_log_dir(kv_norm)

        return f"{model_for_name}-tp{tp_str}-pp{pp_str}-dp{dp_str}-{mns_short}-{mnbt_short}-{spec_short}-{quant_short}-{kv_short}"

    def _expand_experiment(self, experiment_config: dict[str, Any]) -> list[dict[str, Any]]:
        """Expand a single experiment config into multiple runs if any args values are lists.

        If multiple args contain lists, generate the cartesian product (all permutations).
        Each run corresponds to one SLURM job.
        """
        args = experiment_config.get("args", {}) or {}
        experiment_name = experiment_config["name"]
        sweep_keys = [key for key, value in args.items() if isinstance(value, list) and len(value) > 0]

        if not sweep_keys:
            derived_name = self._derive_run_name(args, experiment_name)
            return [
                {
                    "name": derived_name,
                    "experiment": experiment_name,
                    "args": args,
                }
            ]

        ordered_keys = sorted(sweep_keys)
        value_lists = [args[key] for key in ordered_keys]

        expanded_runs: list[dict[str, Any]] = []
        for combo in product(*value_lists):
            new_args = dict(args)
            for key, val in zip(ordered_keys, combo):
                new_args[key] = val

            new_name = self._derive_run_name(new_args, experiment_name)

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
        # Merge fixed_args and run args, with run args overriding fixed_args
        fixed_args = self.config.get("fixed_args", {}) or {}
        var_args = run_config.get("args", {}) or {}
        merged = {**fixed_args, **var_args, "name": run_config["name"]}

        # Prepend experiment name to output-dir for directory grouping
        if "experiment" in run_config and "output-dir" in merged:
            merged["output-dir"] = f"{merged['output-dir']}/{run_config['experiment']}"

        # Convert CLI-style keys (with dashes) to Python kwargs (with underscores)
        return {key.replace("-", "_"): value for key, value in merged.items()}

    def _execute_direct(self, kwargs: dict[str, Any], run_name: str) -> tuple[int, bool]:
        """Execute generate_data.main directly (no subprocess overhead)."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Submitting Slurm job for run: {run_name}")
        logger.info(f"{'=' * 60}")

        if self.dry_run:
            logger.info("[DRY RUN] Slurm job would be submitted")
            return 0, False

        try:
            generate_data_main(**kwargs)
            return 0, False
        except KeyboardInterrupt:
            logger.info(f"\nInterrupted during job submission for run: {run_name}")
            raise
        except Exception as e:
            error_msg = str(e)
            if "Skipping launch" in error_msg or "already been completed" in error_msg:
                logger.info(f"Job skipped: {error_msg}")
                return 0, True
            logger.error(f"Error submitting job for run {run_name}: {e}")
            return 1, False

    def launch(self) -> dict[str, int]:
        """
        Launch all configured experiments as Slurm jobs.

        Returns:
            Dictionary mapping run names to job submission exit codes
        """
        logger.info(f"Configuration: {self.config_path}")
        logger.info(f"Timestamp: {self.timestamp}")

        if self.dry_run:
            logger.info("\n*** DRY RUN MODE - No Slurm jobs will be submitted ***")
        else:
            logger.info("\n*** SUBMITTING SLURM JOBS ***")

        results = {}
        skipped_runs = set()

        # Expand experiments into runs (lists in args produce cartesian product of values)
        expanded_runs: list[dict[str, Any]] = []
        for experiment in self.config["experiments"]:
            expanded_runs.extend(self._expand_experiment(experiment))

        for run_config in expanded_runs:
            run_name = run_config["name"]

            # Build kwargs and call generate_data.main directly (no subprocess overhead)
            kwargs = self._build_kwargs(run_config)
            exit_code, skipped = self._execute_direct(kwargs, run_name)
            results[run_name] = exit_code
            if skipped:
                skipped_runs.add(run_name)

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
            if self.dry_run:
                status = "✅ WOULD SUBMIT" if exit_code == 0 else f"❌ WOULD FAIL ({exit_code})"
            else:
                if run_name in skipped_runs:
                    status = "⏭️ SKIPPED"
                else:
                    status = "✅ SUBMITTED" if exit_code == 0 else f"❌ FAILED ({exit_code})"
            logger.info(f"{run_name:<30} {status}")

        if results:
            successful_submissions = sum(1 for name, code in results.items() if code == 0 and name not in skipped_runs)
            if self.dry_run:
                logger.info(f"\n{successful_submissions}/{len(results)} jobs would be submitted successfully")
            else:
                logger.info(f"\n{successful_submissions}/{len(results)} jobs submitted successfully")
                if skipped_runs:
                    logger.info(f"{len(skipped_runs)} job(s) skipped (already completed)")
                if successful_submissions > 0:
                    logger.info("Use 'squeue -u $USER' to monitor job status")
                    logger.info("Use 'scancel <job_id>' to cancel jobs if needed")

        return results


def main(
    config: str = "examples/inference/benchmark/sample_benchmark_config.yaml",
    dry_run: bool = False,
) -> None:
    try:
        launcher = ExperimentLauncher(
            config_path=config,
            dry_run=dry_run,
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
