#!/usr/bin/env python3
"""
Launch experiments from a YAML configuration file. 
A dry run prints the Slurm submission commands and planned job submissions without actually submitting any jobs or making changes.

Usage:

python dataforge/benchmark/launch_experiments.py --config dataforge/benchmark/sample_benchmark_config.yaml
python dataforge/benchmark/launch_experiments.py --config dataforge/benchmark/sample_benchmark_config.yaml --dry-run
python dataforge/benchmark/launch_experiments.py --config dataforge/benchmark/sample_benchmark_config.yaml --run-names "run1,run3"
"""

import re
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml
from itertools import product
from dataforge.utils import normalize_speculative, encode_spec_segment_for_log_dir
import typer
from datatrove.utils.logging import logger


class ExperimentLauncher:
    """Launches experiments based on YAML configuration."""
    
    def __init__(self, config_path: str, dry_run: bool = False, 
                 selected_runs: list[str] | None = None):
        """
        Initialize the experiment launcher.
        
        Args:
            config_path: Path to YAML configuration file
            dry_run: If True, print commands without executing
            selected_runs: Optional list of specific run names to execute
        """
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.selected_runs = selected_runs or []
        
        # Load and validate configuration
        self.config = self._load_config()
        self._validate_config()
        
        # Setup experiment metadata
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        
    def _load_config(self) -> dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if not config:
            raise ValueError("Empty configuration file")
            
        return config
    
    def _validate_config(self) -> None:
        """Validate the configuration structure."""
        required_keys = ['script', 'runs']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in config: {key}")
        
        # Validate runs structure
        if not isinstance(self.config['runs'], list):
            raise ValueError("'runs' must be a list of dictionaries")
            
        if len(self.config['runs']) == 0:
            raise ValueError("'runs' list cannot be empty")
            
        for i, run in enumerate(self.config['runs']):
            if not isinstance(run, dict):
                raise ValueError(f"Run {i} must be a dictionary")
            if 'name' not in run:
                raise ValueError(f"Run {i} missing required 'name' field")
            # Validate that no run has 'name' in its args (since we auto-generate it)
            if 'name' in run.get('args', {}):
                raise ValueError(
                    f"Run '{run['name']}' should not have 'name' in its args. "
                    f"The run name is automatically passed as --name to the script."
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
        val_str = val_str.replace('/', '-')
        # Keep only safe characters
        val_str = re.sub(r"[^A-Za-z0-9._-]+", "_", val_str)
        # Collapse multiple underscores and trim
        val_str = re.sub(r"_{2,}", "_", val_str).strip('_')
        return val_str
    
    def _derive_run_name(self, args: dict[str, Any], fallback: str) -> str:
        """Derive run name as {model}-tp_{tp}-{spec_short}."""
        model_value = args.get('model-name-or-path') or args.get('model_name_or_path') or fallback
        # Keep hyphens, but avoid slashes in name
        model_for_name = str(model_value).replace('/', '_')
        tp_value = args.get('tp')
        tp_str = str(tp_value) if tp_value is not None else "1"
        spec_raw = args.get('speculative-config') or args.get('speculative_config')
        if isinstance(spec_raw, str) and spec_raw.strip().lower() in ("none", "null", ""):
            spec_raw = None
        spec_norm = normalize_speculative(spec_raw)
        spec_short = encode_spec_segment_for_log_dir(spec_norm)
        return f"{model_for_name}-tp_{tp_str}-{spec_short}"

    def _expand_run(self, run_config: dict[str, Any]) -> list[dict[str, Any]]:
        """Expand a single run config into multiple runs if any args values are lists.

        If multiple args contain lists, generate the cartesian product (all permutations).
        """
        args = run_config.get('args', {}) or {}
        sweep_keys = [key for key, value in args.items() if isinstance(value, list) and len(value) > 0]

        if not sweep_keys:
            derived_name = self._derive_run_name(args, run_config['name'])
            return [{
                'name': derived_name,
                'args': args,
            }]

        ordered_keys = sorted(sweep_keys)
        value_lists = [args[key] for key in ordered_keys]

        expanded_runs: list[dict[str, Any]] = []
        for combo in product(*value_lists):
            new_args = dict(args)
            for key, val in zip(ordered_keys, combo):
                new_args[key] = val

            new_name = self._derive_run_name(new_args, run_config['name'])

            expanded_runs.append({
                'name': new_name,
                'args': new_args,
            })

        return expanded_runs
    
    def _build_command(self, run_config: dict[str, Any]) -> list[str]:
        """
        Build command line from configuration.
        
        Args:
            run_config: Configuration for a specific run
            
        Returns:
            Command as list of strings
        """
        # Build base command - execute script directly
        script = self.config['script']
        if script.endswith('.py'):
            cmd = ['python', script]
        else:
            # Execute script directly (e.g., "rephrase" -> "rephrase")
            cmd = [script]
        
        # Add the run name as --name argument first
        cmd.extend(['--name', run_config['name']])
        
        # Merge fixed_args and run args, with run args overriding fixed_args
        fixed_args = self.config.get('fixed_args', {})
        var_args = run_config.get('args', {})
        merged_args = {**fixed_args, **var_args}
        
        # Add merged arguments to command
        for key, value in merged_args.items():
            self._add_argument_to_command(cmd, key, value)
        
        return cmd
    
    def _add_argument_to_command(self, cmd: list[str], key: str, value: Any) -> None:
        """Add an argument to the command list based on its type."""
        if isinstance(value, bool):
            if value:  # Only add flag if True
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            # Handle multiple values (e.g., --data path1 path2)
            cmd.append(f"--{key}")
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([f"--{key}", str(value)])
    
    def _should_run(self, run_name: str) -> bool:
        """Check if a specific run should be executed."""
        if not self.selected_runs:
            return True
        return run_name in self.selected_runs
    
    def _execute_command(self, cmd: list[str], run_name: str) -> tuple[int, bool]:
        """Execute a command (Slurm job submission) and indicate if it was skipped."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Submitting Slurm job for run: {run_name}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"{'='*60}")
        
        if self.dry_run:
            logger.info("[DRY RUN] Slurm job would be submitted")
            return 0, False
        
        try:
            # Execute the command (which will submit a Slurm job)
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            
            # Print stdout/stderr for job submission feedback
            if result.stdout:
                logger.info(f"Job submission output: {result.stdout.strip()}")
            if result.stderr:
                logger.info(f"Job submission errors: {result.stderr.strip()}")
            combined_output = "\n".join(filter(None, [result.stdout, result.stderr]))
            skipped = "Skipping launch" in combined_output or "already been completed" in combined_output

            if result.returncode != 0:
                logger.error(f"❌ Failed to submit Slurm job for {run_name}")
            
            return result.returncode, skipped
        except KeyboardInterrupt:
            logger.info(f"\nInterrupted during job submission for run: {run_name}")
            raise
        except Exception as e:
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
        
        if self.selected_runs:
            logger.info(f"Selected runs: {', '.join(self.selected_runs)}")
        
        results = {}
        skipped_runs = set()
        
        # Expand runs to handle sweeps (lists in args produce cartesian product of values)
        expanded_runs: list[dict[str, Any]] = []
        for base_run in self.config['runs']:
            expanded_runs.extend(self._expand_run(base_run))

        for run_config in expanded_runs:
            run_name = run_config['name']
            
            if not self._should_run(run_name):
                logger.info(f"Skipping run: {run_name}")
                continue
            
            # Build and submit Slurm job
            cmd = self._build_command(run_config)
            exit_code, skipped = self._execute_command(cmd, run_name)
            results[run_name] = exit_code
            if skipped:
                skipped_runs.add(run_name)
            
            if exit_code != 0:
                logger.error(f"❌ Slurm job submission failed for {run_name} (exit code {exit_code})")
                
                # Check if we should continue on failure
                if not self.config.get('continue_on_failure', True):
                    logger.warning("Stopping job submissions due to failure")
                    break
            elif not self.dry_run:
                if skipped:
                    logger.info(f"ℹ️ Slurm job skipped for {run_name} (already completed)")
                else:
                    logger.info(f"✅ Slurm job submitted successfully for {run_name}")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("JOB SUBMISSION SUMMARY")
        logger.info(f"{'='*60}")
        
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
            successful_submissions = sum(
                1 for name, code in results.items() if code == 0 and name not in skipped_runs
            )
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
    config: str = "dataforge/benchmark/sample_benchmark_config.yaml",
    dry_run: bool = False,
    run_names: str | None = None,
) -> None:
    # Parse selected runs (comma-separated string → list)
    selected_runs: list[str] | None = None
    if run_names:
        selected_runs = [name.strip() for name in run_names.split(",") if name.strip()]

    try:
        launcher = ExperimentLauncher(
            config_path=config,
            dry_run=dry_run,
            selected_runs=selected_runs
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
