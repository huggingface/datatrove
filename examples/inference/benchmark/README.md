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

experiments:
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

Just re-run this script until all jobs complete successfully. Completed jobs and previous failures (OOM, timeout, server_fail) are skipped by default. To re-run only specific failure types:

```bash
# Re-run timeouts and server failures, but still skip OOM
python examples/inference/benchmark/launch_experiments.py \
    --config examples/inference/benchmark/sample_benchmark_config.yaml \
    --skip-failure-reasons OOM

# Re-run all previously failed experiments
python examples/inference/benchmark/launch_experiments.py \
    --config examples/inference/benchmark/sample_benchmark_config.yaml \
    --skip-failure-reasons none
```

### 4. Analyze Results

After jobs complete, analyze the results:

```bash
python examples/inference/benchmark/analyze_results.py \
    --root data
```

This writes per-experiment CSVs plus `benchmarking_results.csv` and `optimization_summary.csv` into the root directory.

## Output Structure

Results are organized under:
```
{root}/{experiment}/{prompt}/{model}/tp{TP}-pp{PP}-dp{DP}/mns_{N}/mnbt_{M}/gmu_{P}/bs_{B}/kvc_{...}/spec_{...}/quant_{...}/inference_logs/
```

Directory segments:
- `{experiment}`: experiment name from config
- `{prompt}`: prompt template name
- `{model}`: model name (org prefix stripped, e.g., `gemma_3_1b_it`)
- `tp{TP}-pp{PP}-dp{DP}`: parallelism config (e.g., `tp2-pp1-dp1`)
- `mns_{N}`: max-num-seqs value (e.g., `mns_256`)
- `mnbt_{M}`: max-num-batched-tokens value (e.g., `mnbt_8192`)
- `gmu_{P}`: gpu-memory-utilization as percentage (e.g., `gmu_90` for 0.9)
- `bs_{N}`: block-size value (e.g., `bs_16`, `bs_32`)
- `kvc_*`: KV cache dtype (see [KV Cache Quantization](#kv-cache-quantization))
- `spec_*`: speculative config (see [Speculative Decoding](#speculative-decoding))
- `quant_*`: quantization config (see [Quantization](#quantization))

## Tiered Optimization Approach

The benchmark follows a sequential two-tier approach where Tier 1 builds on the optimal settings from Tier 0:

| Tier | Parameters              | Goal                                                                |
| ---- | ----------------------- | ------------------------------------------------------------------- |
| 0    | `tp`, `mns`, `mnbt`    | **Parallelism & Batching** - Find optimal parallelism and batch sizes |
| 1    | `spec`, `gmu`          | **SpecDec & GMU** - Lossless speedup via speculative decoding and memory tuning |

Run tiers sequentially and use winners from each tier to inform the next. See [`sample_benchmark_config.yaml`](sample_benchmark_config.yaml) for a complete example.

## Metrics

The analyzer outputs per-experiment CSV files and a combined CSV. Key columns:

| Column                          | Description                                 |
| ------------------------------- | ------------------------------------------- |
| `experiment`, `prompt`          | Experiment and prompt template name         |
| `model`                         | Model name (org prefix stripped)            |
| `tp`, `pp`, `dp`                | Parallelism config                          |
| `mns`, `mnbt`                   | Batching parameters                         |
| `gmu`, `bs`, `kvc`              | Memory and KV cache config                  |
| `spec`                          | Speculative decoding config                 |
| `quant`                         | Quantization config                         |
| `output_tps_per_gpu`            | Output tokens per second per GPU            |
| `gpu_days_to_process_1b_tokens` | GPU-days to generate 1B output tokens       |
| `gpus_for_1b_tokens_per_hour`   | GPUs needed for 1B tokens/hour              |

## Configuration Reference

### Top-Level Keys

| Key                   | Description                                                      |
| --------------------- | ---------------------------------------------------------------- |
| `script`              | Path to the inference script to run                              |
| `continue_on_failure` | If `true`, continue with other runs if one fails                 |
| `fixed_args`          | Arguments applied to all experiments                             |
| `experiments`         | List of experiment configurations (each can spawn multiple runs) |

### Experiment Configuration

Each experiment can specify:
- `name`: Experiment name (top-level directory in output path)
- `args`: Dictionary of arguments; list values are expanded into cartesian product, launching multiple runs per experiment. Experiment args override `fixed_args` for the same key. List values in `fixed_args` are also expanded as sweeps.

## Speculative Decoding

Speculative decoding can significantly improve throughput by speculatively generating multiple tokens in parallel.

### Supported Methods

| Method      | Config                                                        | Description                                      |
| ----------- | ------------------------------------------------------------- | ------------------------------------------------ |
| None        | `None`                                                        | No speculative decoding                          |
| N-gram      | `'{"method": "ngram", "num_speculative_tokens": N}'`          | Uses prompt n-gram matching to speculate tokens  |
| Suffix      | `'{"method": "suffix", "num_speculative_tokens": N}'`         | Uses suffix-based speculation from the prompt    |
| Draft Model | `'{"model": "org/draft-model", "num_speculative_tokens": N}'` | Uses a smaller draft model to speculate tokens   |

> **Note**: Suffix decoding (`'{"method": "suffix", ...}'`) requires the `arctic-inference` package which needs CUDA and GCC 10+ to build. Install manually on GPU nodes if needed: `pip install arctic-inference`

### Example Configuration

```yaml
experiments:
  - name: "gemma-spec-sweep"
    args:
      model-name-or-path: "google/gemma-3-1b-it"
      speculative-config:
        - None  # No speculative decoding
        - '{"method": "ngram", "num_speculative_tokens": 5}'  # N-gram
        - '{"method": "suffix", "num_speculative_tokens": 32}'  # Suffix
        # Draft model (not yet supported as of vLLM 0.15.0):
        # - '{"model": "facebook/opt-125m", "num_speculative_tokens": 5}'
```

### Notes

- **N-gram decoding**: Best for tasks where the output is likely to contain sequences from the prompt (e.g., summarization, extraction)
- **Suffix decoding**: Uses the suffix of the input to speculate tokens; can be effective when outputs share patterns with inputs
- **Draft model**:     Uses a smaller, faster model to speculate tokens that are then verified by the main model; best for maximum throughput when a compatible draft model is available

## Quantization

Reduce memory usage with model quantization:

| Method       | Config           | Description                           |
| ------------ | ---------------- | ------------------------------------- |
| None         | `None`           | No quantization (default)             |
| BitsAndBytes | `"bitsandbytes"` | 4-bit quantization using BitsAndBytes |

## KV Cache Quantization

Reduce KV cache memory with dtype quantization:

| Option   | Config       | Description                                 |
| -------- | ------------ | ------------------------------------------- |
| Auto     | `"auto"`     | Model's default unquantized dtype (default) |
| FP8 E4M3 | `"fp8_e4m3"` | FP8 E4M3 format (CUDA 11.8+)                |
| FP8 E5M2 | `"fp8_e5m2"` | FP8 E5M2 format                             |

## Optimization Results

Results from a two-tier optimization sweep on 80GB H100 GPUs. Baseline uses vLLM defaults: `tp` in (1, 2, 4, 8) (first that fits), `mns=256`, `mnbt=8192`, `gmu=90`, `bs=16`, `kvc=auto`, `spec=none`, `quant=none`.

| Model                       | Base TP | Base tps/gpu | Best tps/gpu | Speedup | Best Parameters                                   |
| :-------------------------- | ------: | -----------: | -----------: | ------: | :------------------------------------------------- |
| SmolLM2-135M-Instruct       |       1 |        28391 |        45540 |   1.60x | mns=512, mnbt=32768, spec=ngram_6                 |
| SmolLM2-360M-Instruct       |       1 |        17887 |        23996 |   1.34x | mns=512, spec=ngram_6                             |
| Qwen3-0.6B                  |       1 |        13527 |        14069 |   1.04x | mns=512                                           |
| gemma-3-270m-it              |       1 |        22996 |        23585 |   1.03x | mnbt=32768                                        |
| gemma-3-1b-it                |       1 |        14838 |        16762 |   1.13x | mns=4096, mnbt=32768                              |
| SmolLM2-1.7B-Instruct       |       1 |         5255 |         9220 |   1.75x | mns=2048, mnbt=32768, gmu=95, spec=suffix_32      |
| Qwen3-1.7B                  |       1 |        11710 |        12313 |   1.05x | mnbt=32768                                        |
| Qwen3-4B                    |       1 |         7919 |         8086 |   1.02x | mnbt=32768                                        |
| gemma-3-4b-it                |       1 |         8501 |         9253 |   1.09x | mns=1024, mnbt=32768                              |
| Qwen3-8B                    |       1 |         6338 |         6443 |   1.02x | gmu=95                                            |
| gemma-3-12b-it               |       1 |         2999 |         3046 |   1.02x | gmu=95                                            |
| Qwen3-14B                   |       1 |         4414 |         4549 |   1.03x | tp=2                                              |
| gpt-oss-20b                 |       1 |        12432 |        14671 |   1.18x | mns=512, mnbt=16384                               |
| gemma-3-27b-it               |       2 |         1724 |         1724 |   1.00x | (baseline is optimal)                             |
| Qwen3-30B-A3B-Thinking-2507 |       1 |         2977 |         5310 |   1.78x | tp=2, mns=512, mnbt=32768                         |
| Qwen3-32B                   |       4 |         1987 |         2078 |   1.05x | mns=512, mnbt=16384, gmu=95                       |
| Qwen3-Next-80B-A3B-Thinking |       4 |         2034 |         2678 |   1.32x | mns=512                                           |
| gpt-oss-120b                |       1 |         3138 |         6117 |   1.95x | tp=2, mns=1024, mnbt=32768                        |

### Key Takeaways

- **Speculative decoding** provides the largest wins for small models (up to 1.75x for SmolLM2-1.7B with suffix decoding, 1.60x for SmolLM2-135M with n-gram)
- **Batch size tuning** (`mns`, `mnbt`) is the most consistently impactful lever across all model sizes
- **MoE models** benefit significantly from TP tuning (gpt-oss-120b: 1.95x with tp=2, Qwen3-30B-A3B: 1.78x with tp=2)
- **Large dense models** (gemma-3-27b) are already well-optimized at baseline defaults

### Environment

| Library      | Version       |
| ------------ | ------------- |
| vLLM         | 0.15.0        |
| PyTorch      | 2.9.1         |
| Transformers | 4.57.6        |
| DataTrove    | 0.8.0         |
| CUDA         | 12.8          |
