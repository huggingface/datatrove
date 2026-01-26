"""Utility functions for the vLLM benchmark scripts."""

import json
from typing import Any

from huggingface_hub import create_repo, get_full_repo_name, repo_exists, whoami

from datatrove.utils.logging import logger


MAX_GPUS_PER_NODE = 8


def check_hf_auth() -> None:
    """
    Checks if the user is authenticated with Hugging Face and has a write token.
    Raises ValueError if not authenticated or if the token is not a write token.
    """
    try:
        user_info = whoami()
        logger.info(f"Authenticated as: {user_info.get('name', 'Unknown')}")
        auth = user_info.get("auth", {})
        if auth.get("type") == "access_token":
            role = auth.get("accessToken", {}).get("role")
            logger.info(f"Token role: {role}")
            if role != "write":
                raise ValueError(
                    "Active token is NOT a write token. Please set HF_TOKEN environment variable to a write token."
                )
    except Exception:
        raise ValueError("Not logged in to Hugging Face. Please set HF_TOKEN environment variable to a write token.")


def resolve_repo_id(output_dataset_name: str) -> str:
    """
    Resolves the full repository ID for the output dataset.
    Handles cases where organization is omitted or explicitly provided.
    """
    try:
        org, model_id = None, output_dataset_name
        if "/" in output_dataset_name:
            org, model_id = output_dataset_name.split("/", 1)
        return get_full_repo_name(model_id=model_id, organization=org)
    except Exception:
        return output_dataset_name


def ensure_repo_exists(repo_id: str, private: bool = True) -> None:
    """
    Ensures that the Hugging Face dataset repository exists.
    If it doesn't exist, it creates it.
    """
    try:
        if not repo_exists(repo_id, repo_type="dataset"):
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=False,
            )
            logger.info(f"HF dataset repo '{repo_id}' created successfully")
    except Exception as e:
        logger.warning(f"Warning: failed to ensure HF dataset repo '{repo_id}' exists: {e}")


def model_name_safe(model: str) -> str:
    """Convert model name to a filesystem-safe string."""
    return model.replace("/", "_").replace("-", "_")


def normalize_speculative(spec) -> str:
    """
    Accepts dict/str/bool and returns a canonical JSON string or empty string.
    Rule: prompt_lookup_max = num_speculative_tokens - 1 (if present).
    Any provided prompt_lookup_max in the input is ignored and recomputed.
    """
    if not spec:
        return ""
    obj = None
    if isinstance(spec, dict):
        obj = dict(spec)
    elif isinstance(spec, str):
        try:
            parsed = json.loads(spec)
            if isinstance(parsed, dict):
                obj = parsed
        except Exception:
            obj = None
    else:
        obj = None

    if isinstance(obj, dict):
        if "num_speculative_tokens" in obj:
            try:
                n = int(obj.get("num_speculative_tokens"))
                obj["prompt_lookup_max"] = max(n - 1, 0)
            except Exception:
                obj.pop("prompt_lookup_max", None)
        return json.dumps(obj, separators=(",", ":"))
    return str(spec)


def encode_spec_segment_for_log_dir(spec_json: str) -> str:
    """
    Encode speculative_config JSON into a stable directory segment:
    - "spec_none" when disabled or missing
    - "spec_ngram_{N}" when method == "ngram" with N tokens (N defaults to 0 if missing)
    """
    if not spec_json:
        return "spec_none"
    try:
        obj = json.loads(spec_json)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        method = str(obj.get("method", "")).lower()
        nst = obj.get("num_speculative_tokens")
        try:
            n_val = int(nst) if nst is not None else 0
        except Exception:
            n_val = 0
        if method in ("", "none", "null"):
            return "spec_none"
        if method == "ngram":
            return f"spec_ngram_{n_val}"
        # For now, only support the requested shapes; treat others as none
        return "spec_none"
    return "spec_none"


def validate_config(
    tp: int,
    pp: int,
    dp: int,
    nodes_per_task: int,
    optimization_level: int,
    config: Any,
    prompt_template: str | None = None,
) -> None:
    """
    Validates configuration parameters for inference.
    Raises ValueError if any configuration is invalid.
    """
    if prompt_template and "[[DOCUMENT]]" not in prompt_template:
        raise ValueError("Prompt template must contain [[DOCUMENT]] variable")

    if tp > MAX_GPUS_PER_NODE:
        logger.warning(
            f"WARNING: tp ({tp}) is greater than MAX_GPUS_PER_NODE ({MAX_GPUS_PER_NODE}). "
            "This is not optimal for performance since it uses slower inter-GPU communication."
        )
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}.")
    if pp < 1:
        raise ValueError(f"pp must be >= 1, got {pp}.")
    if dp < 1:
        raise ValueError(f"dp must be >= 1, got {dp}.")
    if nodes_per_task < 1:
        raise ValueError(f"nodes_per_task must be >= 1, got {nodes_per_task}.")
    if optimization_level not in (0, 1, 2, 3):
        raise ValueError(
            f"optimization_level must be one of (0, 1, 2, 3), got {optimization_level}. 0 has the fastest startup, 3 has the best throughput."
        )

    total_gpus = tp * pp * dp

    if total_gpus > MAX_GPUS_PER_NODE * nodes_per_task:
        raise ValueError(
            f"TPxPPxDP ({tp}x{pp}x{dp}={total_gpus}) is too high. Please set tp/pp/dp to use "
            f"{MAX_GPUS_PER_NODE * nodes_per_task} or fewer GPUs for nodes_per_task={nodes_per_task}."
        )
    if total_gpus % nodes_per_task != 0:
        raise ValueError(
            f"TPxPPxDP ({tp}x{pp}x{dp}={total_gpus}) must be divisible by nodes_per_task ({nodes_per_task})."
        )

    gpus_per_node = total_gpus // nodes_per_task

    if gpus_per_node < 1:
        raise ValueError(f"nodes_per_task ({nodes_per_task}) cannot exceed total GPUs (tp*pp*dp={total_gpus}).")
    if gpus_per_node > MAX_GPUS_PER_NODE:
        raise ValueError(
            f"gpus_per_node ({gpus_per_node}) exceeds GPUS_PER_NODE ({MAX_GPUS_PER_NODE}). Increase nodes_per_task "
            f"(currently {nodes_per_task}) or reduce tp/pp (currently tp={tp}, pp={pp})."
        )

    # Check if tp is valid for vLLM
    # Handle multi-modal configs (e.g., Gemma3) where num_attention_heads is in text_config
    num_heads = int(getattr(config, "num_attention_heads", None) or config.text_config.num_attention_heads)
    if num_heads % tp != 0:
        raise ValueError(
            f"Total number of attention heads ({num_heads}) must be divisible by tensor parallel size (tp={tp})."
        )

    return gpus_per_node
