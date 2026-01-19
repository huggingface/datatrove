# Synthetic Data Generation

## Installation

Install datatrove with inference dependencies:
```sh
pip install datatrove[inference]
```

Make sure to login to your HF account with `hf auth login` using a token with write access since the script creates dataset repos and uploads data.

## Custom Rollouts

This README focuses on `generate_data.py`, a ready-to-use script for prompt-based generation. If you need more control over how generations are orchestrated (e.g., chunked documents, multi-step reasoning, process pools for heavy preprocessing), you can write **custom rollout functions**. See [`inference_chunked.py`](inference_chunked.py) for examples of:
- Simple single-request rollouts
- Chunked rollouts that split long documents and stitch generations together
- CPU-heavy preprocessing with process pools via `shared_context`
- Multi-node distributed inference

## Quickstart

DataTrove provides two main execution modes for generating synthetic training data from existing datasets:

* **Local execution**: Run on a single machine with multiple workers for development and small-scale generation
* **Slurm cluster**: Distribute processing across multiple nodes for large-scale production workloads

The framework reads input datasets from Hugging Face Hub, processes them through a specified language model (supported via the vLLM backend so far), and outputs the generated synthetic data back to Hugging Face Hub with automatic chunking and checkpoint recovery.

### Generate synthetic data locally

For development or small-scale generation, use the `--local-execution` argument to generate data on a single GPU:

```sh
python examples/inference/generate_data.py \
    --input-dataset-name simplescaling/s1K-1.1 \
    --input-dataset-split train \
    --prompt-column question \
    --model-name-or-path Qwen/Qwen3-4B-Thinking-2507 \
    --output-dataset-name s1K-1.1-synthetic \
    --output-dir data \
    --tasks 1 \
    --examples-per-chunk 50 \
    --local-execution
```

Use the `--dp` and `--tp` flags to configure data and tensor parallelism for multi-GPU setups. For example, to run with 2-way tensor parallelism on 2 GPUs, set `--tp 2 --dp 1`.

For multi-node setups, usually you want to set `--tp {NUM_GPUS_PER_NODE}` and then `--pp 2` and `--nodes-per-task 2` to fit a model on 2 nodes.


### Slurm Job Architecture

When running on Slurm, DataTrove automatically manages three separate jobs with inter-dependencies to ensure efficient processing and accurate reporting:

1.  **`inference`** (GPU array job): The main execution job that processes data in parallel using vLLM. It writes Parquet shards directly to the Hugging Face Hub. Upon successful completion of all tasks, it generates a `stats.json` file.
2.  **`monitor`** (CPU job): A lightweight job that periodically polls the repository for progress and updates the dataset card (README.md) with a live progress bar and ETA. The monitor runs in a loop and stops when either `stats.json` is created (inference completed successfully) or the Slurm job disappears from the queue without creating `stats.json` (inference failed/cancelled).
3.  **`datacard`** (CPU job): This job runs only after the successful completion of the inference job (`afterok` dependency). it reads the final `stats.json` and generates the final, comprehensive dataset card with detailed token statistics.

### Generate synthetic data on a Slurm cluster

For large-scale production workloads, distribute the processing across multiple nodes using Slurm:

```sh
python examples/inference/generate_data.py \
    --input-dataset-name simplescaling/s1K-1.1 \
    --input-dataset-split train \
    --prompt-column question \
    --model-name-or-path Qwen/Qwen3-4B-Thinking-2507 \
    --output-dataset-name s1K-1.1-dataforge \
    --output-dir data \
    --num-workers 10 \
    --tasks 20 \
    --examples-per-chunk 50
```

The script will automatically handle chunking, checkpointing, and queue management for you. The `--tasks` flag controls the size of the Slurm array, while `--num-workers` specifies the number of jobs that can run concurrently.

## Input data format

By specifying `--prompt-column`, the script reads that column from the Hugging Face dataset as the input text for each example. Only the prompt column is consumed. Any existing target/label/completion columns are ignored by this script, but you can keep them for evaluation.
It then builds a chat-style request by prepending an optional `--system-prompt` and either using preformatted message lists when present or wrapping the text as a single user message.

Optionally, you can specify `--prompt-template` to define a template that wraps the content from the prompt column. Use the variable `[[DOCUMENT]]` in your template, which will be replaced with the text from the specified column.

### Supported prompt column formats

Set `--prompt-column` to a column that contains one of the following:

- Plain string (single-turn prompt)

  Example dataset row if you use `--prompt-column question`:

  ```json
  {"question": "What color is the sky?"}
  ```

  The request sent to the model becomes:
  - optional system message from `--system-prompt` (if provided)
  - user: the string from `question`

- Plain string with template (single-turn prompt with custom formatting)

  Use `--prompt-template` to wrap the column content in a template. The `[[DOCUMENT]]` variable will be replaced with the text from the specified column.

  Example dataset row from WikiText:

  ```json
  {"text": "The sky appears blue during the day due to Rayleigh scattering..."}
  ```

  The request sent to the model becomes:
  - optional system message from `--system-prompt` (if provided)
  - user: "Summarize the following document: The sky appears blue during the day due to Rayleigh scattering..."

- Messages list (multi-turn chat)

  Provide a list of `{role, content}` objects and set `--prompt-column` to that column (e.g. `messages`). A system message from `--system-prompt` (if provided) is automatically prepended.

  Single-turn:
  ```json
  {"messages": [{"role": "user", "content": "What color is the sky?"}]}
  ```

  Multi-turn:
  ```json
  {
    "messages": [
      {"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "Hello! How can I help?"},
      {"role": "user", "content": "What color is the sky?"}
    ]
  }
  ```
