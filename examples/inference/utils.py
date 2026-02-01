"""Utility functions for the vLLM benchmark scripts."""

import json
from pathlib import Path

from huggingface_hub import create_repo, get_full_repo_name, repo_exists, whoami
from transformers import AutoConfig

from datatrove.utils.logging import logger


MAX_GPUS_PER_NODE = 8

# Valid quantization methods
QUANTIZATION_METHODS = ("bitsandbytes",)


# Valid KV cache dtype options
KV_CACHE_DTYPE_OPTIONS = ("auto", "fp8_e4m3", "fp8_e5m2")


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
    """Convert model name to a filesystem-safe string, stripping the organization prefix."""
    # Strip organization prefix (e.g., "google/gemma-3-1b-it" -> "gemma-3-1b-it")
    if "/" in model:
        model = model.split("/", 1)[1]
    return model.replace("-", "_")


def encode_mns_segment_for_log_dir(max_num_seqs: int) -> str:
    """Encode max_num_seqs into a stable directory segment: 'mns_{value}'."""
    return f"mns_{max_num_seqs}"


def encode_mnbt_segment_for_log_dir(max_num_batched_tokens: int) -> str:
    """Encode max_num_batched_tokens into a stable directory segment: 'mnbt_{value}'."""
    return f"mnbt_{max_num_batched_tokens}"


def encode_bs_segment_for_log_dir(block_size: int) -> str:
    """Encode block_size into a stable directory segment: 'bs_{value}'."""
    return f"bs_{block_size}"


def encode_gmu_segment_for_log_dir(gpu_memory_utilization: float) -> str:
    """Encode gpu_memory_utilization into a stable directory segment: 'gmu_{value}'.

    Uses percentage format for readability (e.g., 0.9 -> 'gmu_90').
    """
    pct = int(gpu_memory_utilization * 100)
    return f"gmu_{pct}"


def normalize_speculative(spec) -> str:
    """
    Accepts dict/str/bool and returns a canonical JSON string or empty string.

    For ngram method: prompt_lookup_max = num_speculative_tokens - 1 (if present).
    For suffix method: no additional parameters are added.
    Any provided prompt_lookup_max in the input is ignored and recomputed for ngram.
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
        method = str(obj.get("method", "")).lower()
        # Only add prompt_lookup_max for ngram method
        if method == "ngram" and "num_speculative_tokens" in obj:
            try:
                n = int(obj["num_speculative_tokens"])
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
    - "spec_suffix_{N}" when method == "suffix" with N tokens (N defaults to 0 if missing)
    - "spec_{model}_{N}" when a draft model is specified (N defaults to 0 if missing)
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

        # Check for draft model speculative decoding (has "model" key for the draft model)
        draft_model = obj.get("model")
        if draft_model and isinstance(draft_model, str):
            # Make the draft model name filesystem-safe
            draft_model_safe = draft_model.replace("/", "_").replace("-", "_")
            return f"spec_{draft_model_safe}_{n_val}"

        if method in ("", "none", "null"):
            return "spec_none"
        if method == "ngram":
            return f"spec_ngram_{n_val}"
        if method == "suffix":
            return f"spec_suffix_{n_val}"
        # Fallback: use the method name directly for future extensibility
        return f"spec_{method}_{n_val}"
    return "spec_none"


def normalize_quantization(quant: str | None) -> str | None:
    """
    Normalize quantization configuration string.

    Args:
        quant: Quantization method string or None

    Returns:
        Normalized quantization string or None if disabled.

    Supported methods:
        - "bitsandbytes": 4-bit quantization using BitsAndBytes
    """
    if quant is None:
        return None
    if isinstance(quant, str):
        quant_lower = quant.strip().lower()
        if quant_lower in ("none", "null", ""):
            return None
        if quant_lower in QUANTIZATION_METHODS:
            return quant_lower
        raise ValueError(f"Unknown quantization method: {quant}. Supported: {QUANTIZATION_METHODS}")
    return None


def encode_quant_segment_for_log_dir(quant: str | None) -> str:
    """
    Encode quantization config into a stable directory segment.

    Returns:
        - "quant_none" when disabled or missing
        - "quant_bnb" for bitsandbytes 4-bit quantization
    """
    if not quant:
        return "quant_none"
    quant_lower = quant.strip().lower()
    if quant_lower == "bitsandbytes":
        return "quant_bnb"
    # Fallback for future methods
    return f"quant_{quant_lower.replace('-', '_')}"


def normalize_kvc_dtype(kv_dtype: str | None) -> str:
    """
    Normalize KV cache dtype configuration string.

    Args:
        kv_dtype: KV cache dtype string or None

    Returns:
        Normalized KV cache dtype string. Defaults to "auto".

    Supported options:
        - "auto": Uses the model's default "unquantized" data type
        - "fp8_e4m3": FP8 E4M3 format (CUDA 11.8+)
        - "fp8_e5m2": FP8 E5M2 format (CUDA 11.8+)
    """
    if kv_dtype is None:
        return "auto"
    if isinstance(kv_dtype, str):
        kv_lower = kv_dtype.strip().lower()
        if kv_lower in ("none", "null", ""):
            return "auto"
        if kv_lower in KV_CACHE_DTYPE_OPTIONS:
            return kv_lower
        raise ValueError(f"Unknown kvc_dtype: {kv_dtype}. Supported: {KV_CACHE_DTYPE_OPTIONS}")
    return "auto"


def encode_kvc_segment_for_log_dir(kv_dtype: str) -> str:
    """
    Encode KV cache dtype config into a stable directory segment (kvc_*).

    Returns:
        - "kvc_auto" for auto (default, unquantized)
        - "kvc_fp8e4m3" for fp8_e4m3
        - "kvc_fp8e5m2" for fp8_e5m2
    """
    kv_lower = kv_dtype.strip().lower() if kv_dtype else "auto"
    if kv_lower in ("auto", "none", "null", ""):
        return "kvc_auto"
    if kv_lower == "fp8_e4m3":
        return "kvc_fp8e4m3"
    if kv_lower == "fp8_e5m2":
        return "kvc_fp8e5m2"
    # Fallback for future options
    return f"kvc_{kv_lower.replace('-', '_')}"


def validate_config(
    tp: int,
    pp: int,
    dp: int,
    nodes_per_task: int,
    optimization_level: int,
    config: AutoConfig,
    prompt_template: str | None = None,
) -> int:
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


def build_run_path(
    output_dir: str,
    prompt_template_name: str,
    model_name_or_path: str,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 8192,
    gpu_memory_utilization: float = 0.9,
    block_size: int = 16,
    kv_cache_dtype: str = "auto",
    speculative_config: str | None = None,
    quantization: str | None = None,
) -> Path:
    """Build the canonical run path for experiment outputs.

    Path structure: {output_dir}/{prompt}/{model}/tp{TP}-pp{PP}-dp{DP}/mns_{N}/mnbt_{M}/gmu_{P}/bs_{B}/kvc_{...}/spec_{...}/quant_{...}
    """
    kv_norm = normalize_kvc_dtype(kv_cache_dtype)
    spec_norm = normalize_speculative(speculative_config) if speculative_config else None
    quant_norm = normalize_quantization(quantization) if quantization else None

    return (
        Path(output_dir)
        / prompt_template_name
        / model_name_safe(model_name_or_path)
        / f"tp{tp}-pp{pp}-dp{dp}"
        / encode_mns_segment_for_log_dir(max_num_seqs)
        / encode_mnbt_segment_for_log_dir(max_num_batched_tokens)
        / encode_gmu_segment_for_log_dir(gpu_memory_utilization)
        / encode_bs_segment_for_log_dir(block_size)
        / encode_kvc_segment_for_log_dir(kv_norm)
        / encode_spec_segment_for_log_dir(spec_norm)
        / encode_quant_segment_for_log_dir(quant_norm)
    )
