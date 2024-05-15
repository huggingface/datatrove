import random
import string
import sys
from datetime import datetime

from loguru import logger

from datatrove.io import DataFolder


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
    logging_dir: DataFolder,
    rank: int,
    local_rank: int = 0,
    colorize_log_files: bool = False,
    colorize_log_output: bool | None = None,
):
    """
    Sets up logging for a given task
    Args:
      logging_dir: DataFolder:
      rank: int:
      local_rank: int:  (Default value = 0)
      colorize_log_files: add colorization to files saved in logs/task_XXXXX.log
      colorize_log_output: add colorization to terminal output logs. None to autodetect

    Returns:

    """
    logger.remove()
    logfile = logging_dir.open(f"logs/task_{rank:05d}.log", "w")
    logger.add(sys.stderr, colorize=colorize_log_output, level="INFO" if local_rank == 0 else "ERROR")
    logger.add(logfile, colorize=colorize_log_files, level="DEBUG")
    logger.info(f"Launching pipeline for {rank=}")
    return logfile


def close_task_logger(logfile, colorize_log_output: bool | None = None):
    """
    Close logfile and reset logging setup
    Args:
      logfile:
      colorize_log_output: add colorization to terminal output logs. None to autodetect

    Returns:

    """
    logger.complete()
    logger.remove()
    reset_logging(colorize_log_output)  # re-add default logger


def reset_logging(colorize_log_output: bool | None = None):
    logger.remove()
    logger.add(sys.stderr, colorize=colorize_log_output)


def log_pipeline(pipeline):
    """
    Print/log pipeline
    Args:
      pipeline:

    Returns:

    """
    steps = "\n".join([pipe.__repr__() if callable(pipe) else "Iterable" for pipe in pipeline])
    logger.info(f"\n--- 🛠️ PIPELINE 🛠\n{steps}")
