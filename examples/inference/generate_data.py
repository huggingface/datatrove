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
    --output-dataset-name s1K-1.1-datatrove \
    --tasks 1 \
    --workers 1 \
    --local-execution

# Generate synthetic data on a Slurm cluster
python examples/inference/benchmark/generate_data.py \
    --input-dataset-name simplescaling/s1K-1.1 \
    --prompt-column question \
    --model-name-or-path Qwen/Qwen3-0.6B \
    --output-dataset-name s1K-1.1-benchmark

# Generate synthetic data using a prompt template with [[DOCUMENT]] variable
python examples/inference/benchmark/generate_data.py \
    --input-dataset-name Salesforce/wikitext \
    --input-dataset-config wikitext-2-v1 \
    --prompt-column text \
    --prompt-template "Summarize the following document: [[DOCUMENT]]" \
    --model-name-or-path Qwen/Qwen3-0.6B \
    --output-dataset-name wikitext-summaries \
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

import typer
from transformers import AutoConfig, GenerationConfig

from datatrove.data import Document
from datatrove.pipeline.inference.dataset_card_generator import (
    InferenceDatasetCardGenerator,
    InferenceDatasetCardParams,
)
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceResult, InferenceRunner
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.utils.logging import logger


# Add parent directory to path so utils can be imported
# This path is also exported in SLURM jobs for unpickling
EXAMPLES_INFERENCE_DIR = str(Path(__file__).parent)
sys.path.insert(0, EXAMPLES_INFERENCE_DIR)
from utils import (  # noqa: E402
    build_run_path,
    check_hf_auth,
    ensure_repo_exists,
    normalize_kvc_dtype,
    normalize_quantization,
    normalize_speculative,
    resolve_repo_id,
    validate_config,
)


MB = 1024 * 1024


def main(
    # Input data details
    input_dataset_name: str = "simplescaling/s1K-1.1",
    input_dataset_config: str | None = None,
    input_dataset_split: str = "train",
    prompt_column: str = "question",
    prompt_template: str | list[str] | None = None,  # Can be "template" or ["name", "template"]
    max_examples: int = -1,
    # Output dataset details
    output_dataset_name: str = "s1K-1.1-datatrove",
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
    # vLLM server settings (there should be no need to change the defaults)
    max_concurrent_generations: int = 500,
    max_concurrent_documents: int = 500,
    max_num_seqs: int = 256,  # reduce this if you run out of memory
    max_num_batched_tokens: int = 8192,  # controls chunked prefill batch size
    gpu_memory_utilization: float = 0.9,  # Fraction of GPU memory for KV cache
    block_size: int = 16,  # KV cache block size (16 or 32)
    speculative_config: str | None = None,
    quantization: str | None = None,  # "bitsandbytes" for 4-bit quantization
    kv_cache_dtype: str = "auto",  # "auto", "fp8_e4m3", or "fp8_e5m2"
    optimization_level: int = 3,  # Set to 0 for fastest startup, 3 for best throughput
    metric_interval: int = 120,
    # Generation parameters
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int = 16384,
    rollouts_per_document: int = 1,
    seed: int | None = None,  # Random seed for reproducible generation
    # Processing settings
    examples_per_chunk: int = 500,
    tasks: int = 10,
    workers: int = 10,
    local_execution: bool = False,
    enable_monitoring: bool = False,
    benchmark_mode: bool = False,  # Skip output writing for benchmarking
    # slurm settings
    name: str = "synth",
    time: str = "12:00:00",
    qos: str = "low",
    reservation: str | None = None,
) -> None:
    """Typer CLI entrypoint that runs the pipeline with provided options."""
    # Skip HuggingFace setup in benchmark mode
    full_repo_id = None
    if benchmark_mode:
        enable_monitoring = False
    else:
        check_hf_auth()  # Check authentication early to avoid errors later
        full_repo_id = resolve_repo_id(output_dataset_name)  # Resolve full repo name for the output dataset
        ensure_repo_exists(full_repo_id, private=output_private)  # Create the repository if it doesn't exist

    if local_execution:
        import torch

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

    # Parse prompt_template: can be "template" or ["name", "template"]
    prompt_template_name, prompt_template = (
        prompt_template if isinstance(prompt_template, list) else ("default", prompt_template)
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

                _logger.warning(
                    f"Truncated content from {original_len} to {len(content)} chars (budget: {char_budget} chars)"
                )

            messages.append({"role": "user", "content": content})

        return await generate(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                **({"seed": seed} if seed is not None else {}),
            }
        )

    generation_config = GenerationConfig.from_pretrained(
        model_name_or_path, revision=model_revision, trust_remote_code=trust_remote_code
    )
    temperature = temperature if temperature is not None else getattr(generation_config, "temperature", 1.0)
    top_p = top_p if top_p is not None else getattr(generation_config, "top_p", 1.0)
    top_k = top_k if top_k is not None else getattr(generation_config, "top_k", -1)

    # Normalize speculative config; treat common "none" strings as disabled
    spec_raw = speculative_config
    if isinstance(spec_raw, str) and spec_raw.strip().lower() in ("none", "null", ""):
        spec_raw = None
    normalized_spec = normalize_speculative(spec_raw)

    # Normalize quantization and KV cache dtype configs
    normalized_quant = normalize_quantization(quantization)
    normalized_kv_dtype = normalize_kvc_dtype(kv_cache_dtype)

    # Build dynamic output directory: {output_dir}/{prompt}/{model}/{tp-pp-dp}/{mns}/{mnbt}/{gmu}/{bs}/{kvc}/{spec}/{quant}
    run_path = build_run_path(
        output_dir=output_dir,
        prompt_template_name=prompt_template_name,
        model_name_or_path=model_name_or_path,
        tp=tp,
        pp=pp,
        dp=dp,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        speculative_config=spec_raw,
        quantization=quantization,
    )
    output_path = f"hf://datasets/{full_repo_id}" if not benchmark_mode else str(run_path / "output")
    checkpoints_path = str(run_path / "checkpoints")
    inference_logs_path = run_path / "inference_logs"
    monitor_logs_path = run_path / "monitor_logs"
    datacard_logs_path = run_path / "datacard_logs"

    # Build quantization-specific kwargs for vLLM
    quant_kwargs: dict[str, Any] = {}
    if normalized_quant == "bitsandbytes":
        # BitsAndBytes 4-bit quantization
        quant_kwargs["quantization"] = "bitsandbytes"

    # Build KV cache dtype kwargs for vLLM
    kv_cache_kwargs: dict[str, Any] = {}
    if normalized_kv_dtype != "auto":
        # FP8 KV cache (reduces memory while maintaining quality)
        kv_cache_kwargs["kv_cache_dtype"] = normalized_kv_dtype
        kv_cache_kwargs["calculate_kv_scales"] = True

    _model_kwargs = {
        "revision": model_revision,
        "dtype": "bfloat16",
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "block-size": block_size,
        "gpu-memory-utilization": gpu_memory_utilization,
        **({"speculative_config": normalized_spec} if normalized_spec else {}),
        **quant_kwargs,
        **kv_cache_kwargs,
        "optimization-level": optimization_level,
    }
    # Memory per CPU for slurm allocation (in GB)
    mem_per_cpu_gb = 22
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
            checkpoints_local_dir=checkpoints_path if not benchmark_mode else None,
            # The HuggingFaceDatasetWriter only uploads at the end, the ParquetWriter uploads incrementally
            output_writer=ParquetWriter(
                output_folder=output_path,
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
            "max_tokens": max_tokens,
            "model_max_context": model_max_context,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
        },
        spec_config=normalized_spec,
        stats_path=str(inference_logs_path / "stats.json"),
    )

    datacard_pipeline = [InferenceDatasetCardGenerator(params=dataset_card_params)]

    if local_execution:
        from datatrove.executor import LocalPipelineExecutor  # Lazy import to speed up startup time

        inference_executor = LocalPipelineExecutor(
            pipeline=inference_pipeline,
            logging_dir=str(inference_logs_path),
            tasks=tasks,
            workers=workers,
        )
        inference_executor.run()

        if not benchmark_mode:
            datacard_executor = LocalPipelineExecutor(
                pipeline=datacard_pipeline,
                logging_dir=str(datacard_logs_path),
                tasks=1,
                workers=1,
            )
            # Monitor not supported in local execution as it would block
            datacard_executor.run()
    else:
        from datatrove.executor import SlurmPipelineExecutor  # Lazy import to speed up startup time

        slurm_env_command = f"source .venv/bin/activate && export PYTHONPATH={EXAMPLES_INFERENCE_DIR}:$PYTHONPATH"

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
            mem_per_cpu_gb=mem_per_cpu_gb,
            # Required so Datatrove starts Ray with GPUs; otherwise it launches Ray with `--num-gpus 0`.
            gpus_per_task=gpus_per_node,
            nodes_per_task=nodes_per_task,
            sbatch_args={
                **({"reservation": reservation} if reservation else {}),
            },
            env_command=slurm_env_command,
        )
        inference_executor.run()

        if enable_monitoring:
            # Lazy import to speed up startup time
            from datatrove.pipeline.inference.progress_monitor import InferenceProgressMonitor

            monitor_pipeline = [
                InferenceProgressMonitor(
                    params=dataset_card_params,
                    max_examples=max_examples,
                    update_interval=60 if local_execution else 3600,  # 1 minute for debugging, 1 hour for slurm
                )
            ]
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
                env_command=slurm_env_command,
            )

            monitor_executor.run()

        if not benchmark_mode:
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
                env_command=slurm_env_command,
            )
            datacard_executor.run()

    return inference_executor.job_id


if __name__ == "__main__":
    typer.run(main)
