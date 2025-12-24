import os
import random
import string
import sys
from datetime import datetime

from loguru import logger


def get_env_bool(name, default=None):
    env_var = os.environ.get(name, None)
    return default if env_var is None else (env_var.lower().strip() in ("yes", "true", "t", "1"))


DATATROVE_COLORIZE_LOGS = get_env_bool("DATATROVE_COLORIZE_LOGS")
DATATROVE_COLORIZE_LOG_FILES = get_env_bool("DATATROVE_COLORIZE_LOG_FILES", False)


def get_timestamp() -> str:
    """
    Get current timestamp as a str
    Returns:

    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_random_str(length=5):
    """
    Random string of a given length with lowercase characters
    Args:
      length:  (Default value = 5)

    Returns:

    """
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def add_task_logger(
    logging_dir,
    rank: int,
    local_rank: int = 0,
    node_rank: int = -1,
):
    """
    Sets up logging for a given task
    Args:
      logging_dir: DataFolder
      rank: int:
      local_rank: int:  (Default value = 0)
      node_rank: int: Node rank for logging prefix. Logs will be prefixed with [NODE X] if node_rank != -1. node_rank == -1 means single node mode (default).
    Returns:

    """
    logger.remove()
    logfile = logging_dir.open(f"logs/task_{rank:05d}.log", "w")

    # Create format string with node prefix at the start (only if node_rank != -1)
    # Format: [NODE X] timestamp | level | module:function:line - message
    node_prefix = f"[NODE {node_rank}] " if node_rank != -1 else ""
    format_string = f"{node_prefix}{{time:YYYY-MM-DD HH:mm:ss.SSS}} | <level>{{level: <8}}</level> | {{name}}:{{function}}:{{line}} - <level>{{message}}</level>"

    logger.add(
        sys.stderr,
        colorize=DATATROVE_COLORIZE_LOGS,
        level="INFO" if local_rank == 0 else "ERROR",
        format=format_string,
    )
    logger.add(
        logfile,
        colorize=DATATROVE_COLORIZE_LOG_FILES,
        level="DEBUG",
        format=format_string,
    )
    logger.info(f"Launching pipeline for {rank=}")
    return logfile


def close_task_logger(logfile):
    """
    Close logfile and reset logging setup
    Args:
      logfile:
    Returns:

    """
    logger.complete()
    setup_default_logger()  # re-add default logger
    logfile.close()


def setup_default_logger():
    logger.remove()
    logger.add(sys.stderr, colorize=DATATROVE_COLORIZE_LOGS)


def log_pipeline(pipeline):
    """
    Print/log pipeline
    Args:
      pipeline:

    Returns:

    """
    steps = "\n".join([pipe.__repr__() if callable(pipe) else "Iterable" for pipe in pipeline])
    logger.info(f"\n--- 🛠️ PIPELINE 🛠\n{steps}")


# set colorization based on env vars
setup_default_logger()
