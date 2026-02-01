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

    def _expand_experiment(self, experiment_config: dict[str, Any]) -> list[dict[str, Any]]:
        """Expand a single experiment config into multiple runs if any args values are lists.

        If multiple args contain lists, generate the cartesian product (all permutations).
        Each run corresponds to one SLURM job.
        """
        fixed_args = self.config.get("fixed_args", {}) or {}
        exp_args = experiment_config.get("args", {}) or {}
        experiment_name = experiment_config["name"]
        sweep_keys = [key for key, value in exp_args.items() if isinstance(value, list) and len(value) > 0]

        if not sweep_keys:
            # Merge fixed_args with exp_args for name derivation (exp_args override fixed_args)
            merged_for_name = {**fixed_args, **exp_args}
            derived_name = self._derive_run_name(merged_for_name)
            return [
                {
                    "name": derived_name,
                    "experiment": experiment_name,
                    "args": exp_args,
                }
            ]

        ordered_keys = sorted(sweep_keys)
        value_lists = [exp_args[key] for key in ordered_keys]

        expanded_runs: list[dict[str, Any]] = []
        for combo in product(*value_lists):
            new_args = dict(exp_args)
            for key, val in zip(ordered_keys, combo):
                new_args[key] = val

            # Merge fixed_args with new_args for name derivation
            merged_for_name = {**fixed_args, **new_args}
            new_name = self._derive_run_name(merged_for_name)

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
