# vLLM Inference Benchmark

This directory contains tools for benchmarking vLLM inference throughput across different models and configurations. It provides scripts to launch sweep experiments via SLURM and analyze the results.

## Overview

The benchmark suite consists of two main tools:

- **`launch_experiments.py`**: Reads a YAML config, expands parameter sweeps (model/TP/speculative decoding), and submits SLURM jobs
- **`analyze_results.py`**: Scans run outputs, parses throughput metrics from server logs, and produces a CSV summary with derived metrics

The benchmark uses [`generate_data.py`](../generate_data.py) (located in the parent directory) as the inference script.

## Quick Start

### 1. Prepare a Configuration

Create a YAML config file specifying the models and configurations to benchmark. See [`sample_benchmark_config.yaml`](sample_benchmark_config.yaml) for an example:

```yaml
script: "examples/inference/generate_data.py"
continue_on_failure: true

fixed_args:
  qos: "high"
  time: "1:00:00"
  model-max-context: 2048
  max-tokens: 1024
  input-dataset-name: "simplescaling/s1K-1.1"
  input-dataset-split: "train"
  prompt-column: "question"
  output-dataset-name: "s1K-1.1-benchmark"
  output-dir: "data"

runs:
  - name: "Qwen3-4B"
    args:
      model-name-or-path: "Qwen/Qwen3-4B-Thinking-2507"
      tp: [1, 2, 4]  # Sweep over TP configurations
      speculative-config: [None]
```

### 2. Dry Run (Preview Commands)

Preview the SLURM commands without submitting:

```bash
python examples/inference/benchmark/launch_experiments.py \
    --config examples/inference/benchmark/sample_benchmark_config.yaml \
    --dry-run
```

### 3. Submit Experiments

Launch the benchmark jobs:

```bash
python examples/inference/benchmark/launch_experiments.py \
    --config examples/inference/benchmark/sample_benchmark_config.yaml
```

Just re-run this script until all jobs complete successfully. Completed jobs will be skipped.

### 4. Analyze Results

After jobs complete, analyze the results:

```bash
python examples/inference/benchmark/analyze_results.py \
    --root data \
    --out-csv data/benchmark_results.csv
```

## Output Structure

Results are expected under:
```
data/{model}/tp{TP}-pp{PP}-dp{DP}/{spec_config}/.../inference/
```

Where:
- `spec_none` indicates no speculative decoding
- `spec_ngram_{N}` indicates N-gram speculative decoding (e.g., `spec_ngram_5`)

## Metrics

The analyzer computes the following metrics:

| Metric | Description |
|--------|-------------|
| `input_tps_per_gpu` | Input tokens per second per GPU |
| `output_tps_per_gpu` | Output tokens per second per GPU |
| `gpu_days_to_process_1b_tokens` | GPU-days required to generate 1B output tokens |
| `node_days_to_process_1b_tokens` | Node-days (8 GPUs) to generate 1B output tokens |
| `gpus_for_1b_tokens_per_hour` | Number of GPUs needed to generate 1B tokens/hour |

## Configuration Reference

### Top-Level Keys

| Key | Description |
|-----|-------------|
| `script` | Path to the inference script to run |
| `continue_on_failure` | If `true`, continue with other runs if one fails |
| `fixed_args` | Arguments applied to all runs |
| `runs` | List of experiment configurations |

### Run Configuration

Each run can specify:
- `name`: Base name for the experiment (auto-derived from model/TP if not unique)
- `args`: Dictionary of arguments; list values are expanded into cartesian product

### Supported Sweep Parameters

- `model-name-or-path`: Model to benchmark
- `tp`: Tensor parallelism configurations (e.g., `[1, 2, 4, 8]`)
- `pp`: Pipeline parallelism (for multi-node)
- `dp`: Data parallelism
- `nodes-per-task`: Number of nodes per task (for multi-node models)
- `speculative-config`: Speculative decoding config (`None`, `"ngram:5"`, etc.)

## Model Lineup

We have benchmarked DataTrove with a variety of models to determine the best throughput and number of GPUs required to 
generate 1 billion tokens per hour. The following table provides an overview of these optimised configurations:


| Model name                                                                                                      | Architecture | Size        | TP | PP | Input TPS/GPU | Output TPS/GPU | GPUs/1B/h |
| :-------------------------------------------------------------------------------------------------------------- | :----------- | :---------- | -: | -: | ------------: | -------------: | --------: |
| [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)                                             | 🧱 Dense      | 🐣 Compact  |  1 |  1 |          2565 |          16616 |        17 |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)                                                       | 🧱 Dense      | 🐣 Compact  |  1 |  1 |          2523 |          15397 |        18 |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)                                             | 🧱 Dense      | 🐣 Compact  |  1 |  1 |           760 |           5427 |        51 |
| [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)                               | 🧱 Dense      | 🐣 Compact  |  1 |  1 |           942 |           7038 |        39 |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)                                                 | 🔀 MoE        | 🦅 Medium   |  1 |  1 |          1330 |           6962 |        40 |
| [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 🔀 MoE        | 🦅 Medium   |  1 |  1 |          1253 |           5490 |        51 |
| [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)   | 🔀 MoE        | 🦅 Medium   |  1 |  1 |          3447 |           9274 |        30 |
| [Qwen/Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)                     | 🔀 MoE        | 🦅 Medium   |  1 |  1 |           483 |           3612 |        77 |
| [Qwen/Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)                     | 🔀 MoE        | 🦅 Medium   |  4 |  1 |           136 |           1017 |       273 |
| [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)                                               | 🔀 MoE        | 🦖 Large    |  2 |  1 |           518 |           2704 |       103 |
| [Qwen/Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507)                 | 🔀 MoE        | 🦖 Large    |  8 |  1 |            32 |            239 |      1161 |
| [moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct)                               | 🔀 MoE        | 🐋 Enormous |  8 |  2 |             5 |             26 |     10645 |

### Architecture

- 🧱 Dense: Standard transformer architecture with all parameters active
- 🔀 MoE:   Mixture of Experts architecture with sparse activation

### Model Sizes (total parameters)

- 🐣 Compact:   <4B parameters
- 🦆 Small:     4B-10B parameters
- 🦅 Medium:    10B-100B parameters
- 🦖 Large:     100B-500B parameters
- 🐋 Enormous:  500B+ parameters
