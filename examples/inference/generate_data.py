"""
Script for generating synthetic data using vLLM inference. Uses the InferenceRunner
with chunking enabled. Documents are processed in chunks with checkpoint support
for resuming from failures. Each chunk is saved to a separate output file.

Supports local execution, SLURM clusters, and multi-node setups.

Usage:

# View all options
python examples/inference/generate_data.py --help

# Generate synthetic data locally using a prompt column
python examples/inference/benchmark/generate_data.py \
    --input-dataset-name simplescaling/s1K-1.1 \
    --prompt-column question \
    --model-name-or-path Qwen/Qwen3-0.6B \
    --output-dataset-name s1K-1.1-dataforge \
    --examples-per-chunk 50 \
    --tasks 1 \
    --workers 1 \
    --local-execution

# Generate synthetic data on a Slurm cluster
python examples/inference/benchmark/generate_data.py \
    --input-dataset-name simplescaling/s1K-1.1 \
    --prompt-column question \
    --model-name-or-path Qwen/Qwen3-0.6B \
    --output-dataset-name s1K-1.1-benchmark \
    --examples-per-chunk 50

# Generate synthetic data using a prompt template with [[DOCUMENT]] variable
python dataforge/generate_data.py \
    --input-dataset-name Salesforce/wikitext \
    --input-dataset-config wikitext-2-v1 \
    --prompt-column text \
    --prompt-template "Summarize the following document: [[DOCUMENT]]" \
    --model-name-or-path Qwen/Qwen3-0.6B \
    --output-dataset-name wikitext-summaries \
    --examples-per-chunk 50 \
    --tasks 1 \
    --workers 1

# Generate synthetic data on multiple nodes
python examples/inference/benchmark/generate_data.py \
    --input-dataset-name simplescaling/s1K-1.1 \
    --prompt-column question \
    --model-name-or-path moonshotai/Kimi-K2-Instruct \
    --model-max-context 1024 \
    --max-tokens 8 \
    --trust-remote-code \
    --output-dataset-name s1K-1.1-benchmark-Kimi-K2-Instruct \
    --examples-per-chunk 10 \
    --tasks 1 \
    --workers 1 \
    --max-examples 100 \
    --nodes-per-task 2 \
    --tp 8 \
    --pp 2 \
    --optimization-level 0 \
    --max-num-seqs=16
"""

import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

import torch
import typer
from transformers import AutoConfig, GenerationConfig

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


sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    check_hf_auth,
    encode_spec_segment_for_log_dir,
    ensure_repo_exists,
    model_name_safe,
    normalize_speculative,
    resolve_repo_id,
    validate_config,
)


MB = 1024 * 1024


def main(
    # Input data details
    input_dataset_name: str = "simplescaling/s1K-1.1",
    input_dataset_split: str = "train",
    input_dataset_config: str | None = None,
    prompt_column: str = "question",
    prompt_template: str | None = None,
    max_examples: int = -1,
    # Output dataset details
    output_dataset_name: str = "s1K-1.1-dataforge",
    output_private: bool = True,
    # Output logs and tmp files
    output_dir: str = "data",
    # Inference settings
    server_type: str = "vllm",
    model_name_or_path: str = "Qwen/Qwen3-0.6B",
    model_revision: str = "main",
    model_max_context: int = 32768,
    system_prompt: str | None = None,
    # WARNING: Set to True only if you trust the model repository.
    # Enabling this allows execution of arbitrary code from the remote repository,
    # which can be a security risk. Use True only for trusted sources.
    trust_remote_code: bool = False,
    # vLLM distribution settings
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    nodes_per_task: int = 1,
    max_num_seqs: int = 1000,  # reduce this if you run out of memory
    # vLLM server settings (there should be no need to change the defaults)
    max_concurrent_generations: int = 500,
    max_concurrent_documents: int = 500,
    metric_interval: int = 120,
    speculative_config: str | None = None,
    optimization_level: int = 3,  # Set to 0 for fastest startup, 3 for best throughput
    # Generation parameters
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int = 16384,
    rollouts_per_document: int = 1,
    # Processing settings
    examples_per_chunk: int = 500,
    tasks: int = 10,
    workers: int = 10,
    local_execution: bool = False,
    enable_datacard: bool = True,
    enable_monitoring: bool = False,
    # slurm settings
    name: str = "dataforge",
    time: str = "12:00:00",
    qos: str = "low",
    reservation: str | None = None,
) -> None:
    """Typer CLI entrypoint that runs the pipeline with provided options."""

    # Check authentication early
    check_hf_auth()  # Check authentication early to avoid errors later

    full_repo_id = resolve_repo_id(output_dataset_name)  # Resolve full repo name for the output dataset

    ensure_repo_exists(full_repo_id, private=output_private)  # Create the repository if it doesn't exist

    if local_execution:
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            raise ValueError("Local execution requires at least one CUDA GPU.")
        tp = min(tp, available_gpus)
        pp = 1
        nodes_per_task = 1
        logger.info(f"Local execution on {available_gpus} GPUs on one node")

    config = AutoConfig.from_pretrained(
        model_name_or_path, revision=model_revision, trust_remote_code=trust_remote_code
    )

    gpus_per_node = validate_config(
        tp=tp,
        pp=pp,
        dp=dp,
        nodes_per_task=nodes_per_task,
        optimization_level=optimization_level,
        config=config,
        prompt_template=prompt_template,
    )

    async def simple_rollout(
        document: Document,
        generate: Callable[[dict[str, Any]], Awaitable[InferenceResult]],
    ) -> InferenceResult:
        """
        Basic rollout that sends a single request per document.

        Returns the InferenceResult directly, which will be stored under document.metadata["rollout_results"].
        """
        messages = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]

        if isinstance(document.text, list) and all(isinstance(msg, dict) for msg in document.text):
            if prompt_template:
                raise ValueError("Prompt template is not supported for message lists")
            messages.extend(document.text)
        else:
            content = prompt_template.replace("[[DOCUMENT]]", document.text) if prompt_template else document.text

            # Truncate content if too long to avoid server errors
            # Uses ~3 chars per token as a conservative approximation
            char_budget = (model_max_context - max_tokens) * 3
            if len(content) > char_budget:
                original_len = len(content)
                # Try to truncate at a newline boundary for cleaner cuts
                last_newline = content.rfind("\n", 0, char_budget)
                content = content[:last_newline] if last_newline != -1 else content[:char_budget]
                # Import logger inside the function to ensure it's available in pickled closures
                from datatrove.utils.logging import logger as _logger
                _logger.warning(f"Truncated content from {original_len} to {len(content)} chars (budget: {char_budget} chars)")

            messages.append({"role": "user", "content": content})

        return await generate(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        )

    generation_config = GenerationConfig.from_pretrained(
        model_name_or_path, revision=model_revision, trust_remote_code=trust_remote_code
    )
    temperature = temperature if temperature is not None else getattr(generation_config, "temperature", 1.0)
    top_p = top_p if top_p is not None else getattr(generation_config, "top_p", 1.0)
    top_k = top_k if top_k is not None else getattr(generation_config, "top_k", -1)

    # Normalize and encode speculative config; treat common "none" strings as disabled
    spec_raw = speculative_config
    if isinstance(spec_raw, str) and spec_raw.strip().lower() in ("none", "null", ""):
        spec_raw = None
    normalized_spec = normalize_speculative(spec_raw)
    spec_short = encode_spec_segment_for_log_dir(normalized_spec)

    # Build dynamic output directory: base/modelname/tp-pp-dp/spec_short
    model_dir = model_name_safe(model_name_or_path)
    final_output_dir = os.path.join(output_dir, model_dir, f"tp{tp}-pp{pp}-dp{dp}", spec_short)
    logs_dir = os.path.join(final_output_dir, "logs")
    inference_logs_path = os.path.join(logs_dir, "inference")
    monitor_logs_path = os.path.join(logs_dir, "monitor")
    datacard_logs_path = os.path.join(logs_dir, "datacard")
    checkpoints_path = os.path.join(final_output_dir, "checkpoints")

    _model_kwargs = {
        "revision": model_revision,
        "dtype": "bfloat16",
        "max_num_seqs": max_num_seqs,
        "optimization-level": optimization_level,
        **({"speculative_config": normalized_spec} if normalized_spec else {}),
    }
    # Datatrove's distributed Ray helpers interpret DATATROVE_MEM_PER_CPU as **MB**
    # (despite the executor naming it `mem_per_cpu_gb`). Keep the sbatch directive in MB
    # and also pass the same MB value to the executor so Ray object store sizing is sane.
    mem_per_cpu_mb = 22545
    if not local_execution and nodes_per_task > 1:
        # vLLM defaults to the mp backend when TP fits on a single host; but when TP spans
        # multiple nodes we must force the Ray backend so TP can exceed local GPU count.
        _model_kwargs["distributed-executor-backend"] = "ray"
        # Help any Ray client in subprocesses (like `vllm serve`) attach to the running cluster.
        os.environ["RAY_ADDRESS"] = "auto"

    inference_config = InferenceConfig(
        server_type=server_type,
        model_name_or_path=model_name_or_path,
        model_kwargs=_model_kwargs,
        model_max_context=model_max_context,
        rollouts_per_document=rollouts_per_document,
        max_concurrent_generations=max_concurrent_generations,
        max_concurrent_documents=max_concurrent_documents,
        metric_interval=metric_interval,
        tp=tp,
        dp=dp,
        pp=pp,
        server_log_folder=str(inference_logs_path / "server_logs"),
    )

    inference_pipeline = [
        HuggingFaceDatasetReader(
            dataset=input_dataset_name,
            dataset_options={"name": input_dataset_config, "split": input_dataset_split},
            text_key=prompt_column,
            limit=max_examples,
        ),
        InferenceRunner(
            rollout_fn=simple_rollout,
            config=inference_config,
            records_per_chunk=examples_per_chunk,
            checkpoints_local_dir=checkpoints_path,
            output_writer=ParquetWriter(  # The HuggingFaceDatasetWriter only uploads at the end, the ParquetWriter uploads incrementally
                output_folder=f"hf://datasets/{full_repo_id}",
                output_filename="data/${rank}_${chunk_index}.parquet",
                expand_metadata=True,
                max_file_size=MB if local_execution else 256 * MB,  # ~1MB for debugging, ~256MB for slurm
                batch_size=10 if local_execution else 1000,
            ),
        ),
    ]

    dataset_card_params = InferenceDatasetCardParams(
        output_repo_id=full_repo_id,
        input_dataset_name=input_dataset_name,
        input_dataset_split=input_dataset_split,
        input_dataset_config=input_dataset_config,
        prompt_column=prompt_column,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        model_name=model_name_or_path,
        model_revision=model_revision,
        generation_kwargs={
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "model_max_context": model_max_context,
        },
        spec_config=normalized_spec,
        stats_path=str(inference_logs_path / "stats.json"),
    )

    monitor_pipeline = [
        InferenceProgressMonitor(
            params=dataset_card_params,
            max_examples=max_examples,
            update_interval=60 if local_execution else 3600,  # 1 minute for debugging, 1 hour for slurm
        )
    ]

    datacard_pipeline = [InferenceDatasetCardGenerator(params=dataset_card_params)]

    if local_execution:
        inference_executor = LocalPipelineExecutor(
            pipeline=inference_pipeline,
            logging_dir=str(inference_logs_path),
            tasks=tasks,
            workers=workers,
        )
        inference_executor.run()

        if enable_datacard:
            datacard_executor = LocalPipelineExecutor(
                pipeline=datacard_pipeline,
                logging_dir=str(datacard_logs_path),
                tasks=1,
                workers=1,
            )
            # Monitor not supported in local execution as it would block
            datacard_executor.run()
    else:
        inference_executor = SlurmPipelineExecutor(
            pipeline=inference_pipeline,
            logging_dir=str(inference_logs_path),
            tasks=tasks,
            workers=workers,
            time=time,
            partition="hopper-prod",
            max_array_launch_parallel=True,
            qos=qos,
            job_name=f"{name}_inference",
            cpus_per_task=gpus_per_node * 11,
            # NOTE: Datatrove uses this value to size Ray object store memory (in MB).
            mem_per_cpu_gb=mem_per_cpu_mb,
            # Required so Datatrove starts Ray with GPUs; otherwise it launches Ray with `--num-gpus 0`.
            gpus_per_task=gpus_per_node,
            nodes_per_task=nodes_per_task,
            sbatch_args={
                "mem-per-cpu": f"{mem_per_cpu_mb}M",
                **({"reservation": reservation} if reservation else {}),
            },
            venv_path=".venv/bin/activate",
        )
        inference_executor.run()

        if enable_monitoring:
            # Update monitor with inference job id so it can stop if inference fails
            monitor_pipeline[0].inference_job_id = inference_executor.job_id

            monitor_executor = SlurmPipelineExecutor(
                pipeline=monitor_pipeline,
                logging_dir=str(monitor_logs_path),
                tasks=1,
                workers=1,
                time="7-00:00:00",  # Long enough to outlast inference
                partition="hopper-cpu",
                qos=qos,
                job_name=f"{name}_monitor",
                cpus_per_task=1,
                sbatch_args={"mem-per-cpu": "4G", "requeue": ""},  # Requeue to handle long running jobs
                venv_path=".venv/bin/activate",
            )

            monitor_executor.run()

        if enable_datacard:
            datacard_executor = SlurmPipelineExecutor(
                pipeline=datacard_pipeline,
                logging_dir=str(datacard_logs_path),
                tasks=1,
                workers=1,
                time="0:10:00",
                partition="hopper-cpu",
                qos=qos,
                job_name=f"{name}_datacard",
                cpus_per_task=1,
                depends=inference_executor,
                run_on_dependency_fail=False,  # use afterok
                sbatch_args={"mem-per-cpu": "4G"},
                venv_path=".venv/bin/activate",
            )
            datacard_executor.run()


if __name__ == "__main__":
    typer.run(main)
