"""File-based locking to prevent concurrent torch.compile cache corruption.

When multiple SLURM jobs with the same vLLM config start simultaneously, they may
all try to compile torch.compile graphs at the same time, corrupting the shared cache.
This module provides a locking mechanism to serialize compilation.
"""

import fcntl
import hashlib
import os
from pathlib import Path

from loguru import logger


# Directory for compilation lock files - must be on a shared filesystem visible to all jobs
# Using home directory instead of /tmp because /tmp may be per-job on some SLURM configurations
DEFAULT_LOCK_DIR = Path(os.environ.get("VLLM_COMPILE_LOCK_DIR", os.path.expanduser("~/.cache/vllm_compile_locks")))


class CompileLockManager:
    """Manages file-based locking to prevent concurrent vLLM compilation.

    Usage:
        lock_mgr = CompileLockManager(config_hash)
        if lock_mgr.acquire():
            # We need to compile - lock is held
            try:
                ... # compilation happens here
                lock_mgr.mark_complete()
            finally:
                lock_mgr.release()
        else:
            # Cache already exists, no compilation needed
            pass
    """

    def __init__(self, config_hash: str, lock_dir: Path | None = None):
        """
        Initialize the lock manager.

        Args:
            config_hash: Unique identifier for this configuration
            lock_dir: Directory for lock files (defaults to DEFAULT_LOCK_DIR)
        """
        lock_dir = lock_dir or DEFAULT_LOCK_DIR
        self._config_hash = config_hash
        self.lock_path = lock_dir / f"vllm_compile_{config_hash}.lock"
        self.marker_path = self.lock_path.with_suffix(".done")
        self._lock_file: int | None = None

    def cache_exists(self) -> bool:
        """Check if compilation cache already exists (indicated by .done marker)."""
        return self.marker_path.exists()

    def acquire(self) -> bool:
        """Acquire an exclusive lock for compilation.

        Returns True if lock was acquired (compilation needed), False if cache
        already exists (no compilation needed).
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Fast path: if cache already exists, no need to lock
        if self.cache_exists():
            logger.info(f"Compilation cache exists for config {self._config_hash}, skipping lock")
            return False

        logger.info(f"Acquiring compilation lock: {self.lock_path}")
        self._lock_file = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
        fcntl.flock(self._lock_file, fcntl.LOCK_EX)
        logger.info("Compilation lock acquired")

        # Double-check after acquiring lock (another job may have finished)
        if self.cache_exists():
            logger.info("Cache was created while waiting for lock, releasing immediately")
            self.release()
            return False
        return True

    def mark_complete(self) -> None:
        """Mark compilation as complete by creating the .done marker file."""
        self.marker_path.touch()
        logger.info(f"Marked compilation cache as complete: {self.marker_path}")

    def release(self) -> None:
        """Release the compilation lock."""
        if self._lock_file is not None:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            os.close(self._lock_file)
            self._lock_file = None
            logger.info("Compilation lock released")


def _get_vllm_version() -> str:
    """Get vLLM version string, or 'unknown' if not installed."""
    try:
        from vllm import __version__

        return __version__
    except ImportError:
        return "unknown"


def compute_config_hash(
    model_name_or_path: str,
    tp: int,
    pp: int,
    dp: int,
    model_max_context: int,
    model_kwargs: dict | None = None,
) -> str:
    """Compute a hash of vLLM config parameters for cache/lock identification.

    This hash covers the same key factors as vLLM's VllmConfig.compute_hash():
    - vLLM version (compilation may differ between versions)
    - Model name/path (determines architecture)
    - Parallelism settings (tp, pp, dp affect graph sharding)
    - Max context length (affects tensor shapes)
    - Model kwargs (captures dtype, quantization, etc. if specified)

    Args:
        model_name_or_path: Model name or path
        tp: Tensor parallelism
        pp: Pipeline parallelism
        dp: Data parallelism
        model_max_context: Maximum context length
        model_kwargs: Additional model keyword arguments (e.g., dtype, quantization)

    Returns:
        12-character hex hash string
    """
    config_parts = [
        _get_vllm_version(),
        model_name_or_path,
        str(tp),
        str(pp),
        str(dp),
        str(model_max_context),
    ]
    if model_kwargs:
        for k, v in sorted(model_kwargs.items()):
            config_parts.append(f"{k}={v}")
    config_str = "|".join(config_parts)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]
