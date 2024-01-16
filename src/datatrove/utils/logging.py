import random
import string
import sys
from datetime import datetime

from loguru import logger

from datatrove.io import DataFolder


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_random_str(length=5):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def add_task_logger(logging_dir: DataFolder, rank: int, local_rank: int = 0):
    logger.remove()
    logging_dir.makedirs("logs", exist_ok=True)
    logfile = logging_dir.open(f"logs/task_{rank:05d}.log", "w")
    logger.add(sys.stderr, level="INFO" if local_rank == 0 else "ERROR")
    logger.add(logfile, colorize=True, level="DEBUG")
    logger.info(f"Launching pipeline for {rank=}")
    return logfile


def close_task_logger(logfile):
    logger.complete()
    logger.remove()
    logfile.close()
    logger.add(sys.stderr)  # re-add default logger


def log_pipeline(pipeline):
    steps = "\n".join([pipe.__repr__() if callable(pipe) else "Iterable" for pipe in pipeline])
    logger.info(f"\n--- 🛠️ PIPELINE 🛠\n{steps}")
