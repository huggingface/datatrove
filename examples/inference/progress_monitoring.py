"""
Progress monitoring example for synthetic data generation.

This example demonstrates how to use the InferenceProgressMonitor to track
generation progress and automatically update a HuggingFace dataset card with
a progress bar and ETA. After inference completes, InferenceDatasetCardGenerator
creates a final dataset card with statistics.

Usage:
    # Local execution (requires GPU)
    python examples/inference/progress_monitoring.py --output-dataset-name my-dataset --local-execution

    # Slurm execution with progress monitoring
    python examples/inference/progress_monitoring.py --output-dataset-name my-dataset --enable-monitoring

    # Slurm execution without progress monitoring
    python examples/inference/progress_monitoring.py --output-dataset-name my-dataset
"""

import argparse
import os
from typing import Any, Awaitable, Callable

from huggingface_hub import create_repo, get_full_repo_name, repo_exists, whoami

from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.inference.dataset_card_generator import (
    InferenceDatasetCardGenerator,
    InferenceDatasetCardParams,
)
from datatrove.pipeline.inference.progress_monitor import InferenceProgressMonitor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceResult, InferenceRunner
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.utils.logging import logger


# =============================================================================
# Hardcoded configuration - modify these for your use case
# =============================================================================
INPUT_DATASET = "simplescaling/s1K-1.1"
INPUT_SPLIT = "train"
PROMPT_COLUMN = "question"
MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 2048
EXAMPLES_PER_CHUNK = 500
OUTPUT_DIR = "data"


# =============================================================================
# Helper functions
# =============================================================================
def check_hf_auth() -> None:
    """Check if authenticated with HuggingFace with a write token."""
    try:
        user_info = whoami()
        logger.info(f"Authenticated as: {user_info.get('name', 'Unknown')}")
        auth = user_info.get("auth", {})
        if auth.get("type") == "access_token":
            role = auth.get("accessToken", {}).get("role")
            if role != "write":
                raise ValueError("Token is not a write token. Set HF_TOKEN to a write token.")
    except Exception as e:
        raise ValueError(f"Not logged in to HuggingFace: {e}. Set HF_TOKEN environment variable.")


def resolve_repo_id(output_dataset_name: str) -> str:
    """Resolve full repo ID, adding username if not provided."""
    org, model_id = None, output_dataset_name
    if "/" in output_dataset_name:
        org, model_id = output_dataset_name.split("/", 1)
    return get_full_repo_name(model_id=model_id, organization=org)


def ensure_repo_exists(repo_id: str) -> None:
    """Create HuggingFace dataset repo if it doesn't exist."""
    if not repo_exists(repo_id, repo_type="dataset"):
        create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=False)
        logger.info(f"Created HF dataset repo: {repo_id}")


# =============================================================================
# Rollout function
# =============================================================================
async def simple_rollout(
    document: Document,
    generate: Callable[[dict[str, Any]], Awaitable[InferenceResult]],
) -> InferenceResult:
    """Basic rollout that sends a single request per document."""
    # Note: Using hardcoded value instead of global MAX_TOKENS because globals
    # aren't captured when the function is pickled for Slurm execution
    return await generate(
        {
            "messages": [{"role": "user", "content": document.text}],
            "max_tokens": 2048,
        }
    )


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data with progress monitoring")
    parser.add_argument(
        "--output-dataset-name",
        type=str,
        required=True,
        help="Output HuggingFace dataset name (e.g., 'my-dataset' or 'username/my-dataset')",
    )
    parser.add_argument(
        "--local-execution",
        action="store_true",
        help="Run locally instead of on Slurm (requires GPU)",
    )
    parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        help="Enable progress monitoring (Slurm only, updates dataset card periodically)",
    )
    args = parser.parse_args()

    # Check authentication and resolve repo
    check_hf_auth()
    full_repo_id = resolve_repo_id(args.output_dataset_name)
    ensure_repo_exists(full_repo_id)
    logger.info(f"Output dataset: https://huggingface.co/datasets/{full_repo_id}")

    # Setup paths
    model_safe = MODEL.replace("/", "_")
    final_output_dir = os.path.join(OUTPUT_DIR, model_safe)
    logs_dir = os.path.join(final_output_dir, "logs")
    inference_logs_path = os.path.join(logs_dir, "inference")
    monitor_logs_path = os.path.join(logs_dir, "monitor")
    datacard_logs_path = os.path.join(logs_dir, "datacard")
    checkpoints_path = os.path.join(final_output_dir, "checkpoints")
    stats_path = os.path.join(inference_logs_path, "stats.json")

    # Dataset card parameters (shared between monitor and generator)
    dataset_card_params = InferenceDatasetCardParams(
        output_repo_id=full_repo_id,
        input_dataset_name=INPUT_DATASET,
        input_dataset_split=INPUT_SPLIT,
        input_dataset_config=None,
        prompt_column=PROMPT_COLUMN,
        prompt_template=None,
        system_prompt=None,
        model_name=MODEL,
        model_revision="main",
        generation_kwargs={"max_tokens": MAX_TOKENS},
        spec_config=None,
        stats_path=stats_path,
    )

    # Inference pipeline
    inference_pipeline = [
        HuggingFaceDatasetReader(
            dataset=INPUT_DATASET,
            dataset_options={"split": INPUT_SPLIT},
            text_key=PROMPT_COLUMN,
        ),
        InferenceRunner(
            rollout_fn=simple_rollout,
            config=InferenceConfig(
                server_type="vllm",
                model_name_or_path=MODEL,
                model_kwargs={
                    "max_num_seqs": 500,
                    "enforce-eager": True,
                },  # enforce-eager avoids compile cache conflicts
                server_log_folder=os.path.join(inference_logs_path, "server_logs"),
            ),
            records_per_chunk=EXAMPLES_PER_CHUNK,
            checkpoints_local_dir=checkpoints_path,
            output_writer=ParquetWriter(  # Streams to HF for real-time progress
                output_folder=f"hf://datasets/{full_repo_id}",
                output_filename="data/${rank}_${chunk_index}.parquet",
                expand_metadata=True,
                max_file_size=1024 * 1024,  # ~1MB so we can see progress in real time
                batch_size=10,
            ),
        ),
    ]

    # Monitor pipeline (updates progress to dataset card)
    monitor_pipeline = [
        InferenceProgressMonitor(
            params=dataset_card_params,
            update_interval=60,  # Every minute so we can see progress in real time
        )
    ]

    # Dataset card pipeline (generates final card after inference)
    datacard_pipeline = [InferenceDatasetCardGenerator(params=dataset_card_params)]

    if args.local_execution:
        # Local execution
        inference_executor = LocalPipelineExecutor(
            pipeline=inference_pipeline,
            logging_dir=inference_logs_path,
            tasks=1,
        )
        datacard_executor = LocalPipelineExecutor(
            pipeline=datacard_pipeline,
            logging_dir=datacard_logs_path,
            tasks=1,
        )

        logger.info("Running inference locally...")
        inference_executor.run()
        logger.info("Generating dataset card...")
        datacard_executor.run()
        logger.info(f"Done! Check: https://huggingface.co/datasets/{full_repo_id}")
    else:
        # Slurm execution
        inference_executor = SlurmPipelineExecutor(
            pipeline=inference_pipeline,
            logging_dir=inference_logs_path,
            tasks=10,
            workers=4,
            cpus_per_task=11,
            gpus_per_task=1,
            nodes_per_task=1,
            time="12:00:00",
            partition="hopper-prod",
            job_name="inference",
            qos="normal",
            venv_path=".venv/bin/activate",
        )
        inference_executor.run()

        if args.enable_monitoring:
            # Update monitor with inference job ID to stop if inference fails
            monitor_pipeline[0].inference_job_id = inference_executor.job_id

            monitor_executor = SlurmPipelineExecutor(
                pipeline=monitor_pipeline,
                logging_dir=monitor_logs_path,
                tasks=1,
                time="7-00:00:00",
                partition="hopper-cpu",
                job_name="monitor",
                qos="low",
                venv_path=".venv/bin/activate",
            )
            monitor_executor.run()
            logger.info(f"Monitor job submitted: {monitor_executor.job_id}")

        datacard_executor = SlurmPipelineExecutor(
            pipeline=datacard_pipeline,
            logging_dir=datacard_logs_path,
            tasks=1,
            time="00:10:00",
            partition="hopper-cpu",
            depends=inference_executor,
            job_name="datacard",
            qos="low",
            venv_path=".venv/bin/activate",
        )
        datacard_executor.run()

        logger.info("Jobs submitted!")
        logger.info(f"  Inference job: {inference_executor.job_id}")
        logger.info(f"  Datacard job: {datacard_executor.job_id}")
        logger.info(f"Check: https://huggingface.co/datasets/{full_repo_id}")


if __name__ == "__main__":
    main()
