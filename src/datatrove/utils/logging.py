import random
import string
import sys
from datetime import datetime

from loguru import logger

from datatrove.io import BaseOutputDataFolder


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_random_str(length=5):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def add_task_logger(logging_dir: BaseOutputDataFolder, rank: int, local_rank: int = 0):
    logger.remove()
    logger.add(sys.stderr, level="INFO" if local_rank == 0 else "ERROR")
    logger.add(logging_dir.open(f"logs/task_{rank:05d}.log"), colorize=True, level="DEBUG")
    logger.info(f"Launching pipeline for {rank=}")


def close_task_logger(logging_dir: BaseOutputDataFolder, rank: int):
    logger.remove()
    logging_dir.open(f"logs/task_{rank:05d}.log").close()
