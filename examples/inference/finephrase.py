#!/usr/bin/env python3
"""
Launch FinePhrase production runs via ``generate_data``.

This standalone script calls ``generate_data.main`` once per prompt template to
generate a synthetic dataset at scale.

This script uses the best configuration found from the two-tier benchmark optimization:
   tp=1, mns=2048, mnbt=32768, gmu=0.95, spec=suffix_32 (1.75x speedup)

Usage:
    python examples/inference/finephrase.py
"""

from typing import Any

from generate_data import main as generate_data_main

from datatrove.utils.logging import logger


KWARGS: dict[str, Any] = {
    "time": "7-00:00:00",
    "qos": "low",
    "model_name_or_path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "model_max_context": 8192,
    "max_tokens": 2048,  # Could do 4096, but the model often does not generate as much anyway
    "input_dataset_name": "HuggingFaceFW/fineweb-edu",
    "input_dataset_config": "sample-350BT",
    "input_dataset_split": "train",
    "prompt_column": "text",
    "output_dataset_name": "HuggingFaceFW/finephrase",
    "output_dir": "finephrase",
    "output_private": False,
    "max_concurrent_generations": 5000,
    "max_concurrent_documents": 5000,
    "max_num_seqs": 2048,
    "max_num_batched_tokens": 16384,  # Optimized for 32768 tokens, but we use 16384 tokens to avoid OOM
    "gpu_memory_utilization": 0.90,  # Optimized one is 0.95, but this leads to OOM sometimes
    "speculative_config": '{"method": "suffix", "num_speculative_tokens": 32}',
    "enable_monitoring": True,
    "examples_per_chunk": 100_000,  # 500 are around 2MB => 100_000 are around 400MB (~270MB compressed)
    "optimization_level": 3,  # Set to 0 for debugging
    "workers": 100,
    "tasks": 100,
}

PROMPT_TEMPLATES: dict[str, str] = {
    "math": (
        "Rewrite the document to create a mathematical word problem based on the numerical data or relationships in "
        "the text. Provide a step-by-step solution that shows the calculation process clearly. Create a problem that "
        "requires multi-step reasoning and basic arithmetic operations. It should include the question followed by a "
        "detailed solution showing each calculation step. Output only the problem and solution, nothing else.\n\n"
        "Document: [[DOCUMENT]]"
    ),
    "table": (
        "Rewrite the document as a structured table that organizes the key information, then generate one "
        "question-answer pair based on the table. First extract the main data points and organize them into a clear "
        "table format with appropriate headers using markdown table syntax with proper alignment. After the table, "
        "generate one insightful question that can be answered using the table data. Provide a clear, concise answer "
        "to the question based on the information in the table. Output only the table followed by the question-answer "
        "pair, nothing else.\n\nDocument: [[DOCUMENT]]"
    ),
    "faq": (
        "Rewrite the document as a comprehensive FAQ (Frequently Asked Questions). Extract or infer the key questions "
        "a reader would have about this topic, then provide clear, direct answers. Order questions logically, from "
        "foundational to advanced, or by topic area. Each answer should be self-contained and understandable without "
        "reference to other answers. Ensure the FAQ works as a standalone document. Output only the FAQ, nothing "
        "else.\n\nDocument: [[DOCUMENT]]"
    ),
    "tutorial": (
        "Rewrite the document as a clear, step-by-step tutorial or instructional guide. Use numbered steps or bullet "
        "points where appropriate to enhance clarity. Preserve all essential information while ensuring the style "
        "feels didactic and easy to follow. Output only the tutorial, nothing else.\n\nDocument: [[DOCUMENT]]"
    ),
}


def launch_template(template_name: str, template_text: str) -> str | None:
    """Launch one FinePhrase generation run for a prompt template.

    Args:
        template_name: Prompt template identifier (used in output path and run name).
        template_text: Prompt template containing the ``[[DOCUMENT]]`` placeholder.

    Returns:
        Submitted Slurm job ID if available, otherwise ``None``.
    """
    run_name = f"finephrase_{template_name}"
    logger.info(f"Launching FinePhrase run '{run_name}'")
    job_id = generate_data_main(
        **KWARGS,
        name=run_name,
        prompt_template=[template_name, template_text],
    )
    if job_id is None:
        return None
    return str(job_id)


def main() -> None:
    """Launch all FinePhrase prompt-template runs."""
    job_ids: list[str] = []
    for prompt_template in PROMPT_TEMPLATES.items():
        job_id = launch_template(*prompt_template)
        if job_id is not None:
            job_ids.append(job_id)

    logger.info(f"Launched {len(PROMPT_TEMPLATES)} FinePhrase runs.")
    if job_ids:
        logger.info(f"Submitted job IDs: {' '.join(job_ids)}")


if __name__ == "__main__":
    main()
