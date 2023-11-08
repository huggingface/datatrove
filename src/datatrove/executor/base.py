from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from typing import Callable

from loguru import logger

from datatrove.io import BaseOutputDataFolder, LocalOutputDataFolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import add_task_logger, get_random_str, get_timestamp
from datatrove.utils.stats import PipelineStats


class PipelineExecutor(ABC):
    @abstractmethod
    def __init__(self, pipeline: list[PipelineStep | Callable], logging_dir: BaseOutputDataFolder = None):
        self.pipeline: list[PipelineStep | Callable] = pipeline
        self.logging_dir = (
            logging_dir if logging_dir else LocalOutputDataFolder(f"logs/{get_timestamp()}_{get_random_str()}")
        )

        pipeline = "\n".join([pipe.__repr__() if callable(pipe) else "Sequence..." for pipe in self.pipeline])
        print(f"--- 🛠️PIPELINE 🛠\n{pipeline}")

    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def world_size(self):
        return 0

    def _run_for_rank(self, rank: int, local_rank: int = -1) -> PipelineStats:
        if self.is_rank_completed(rank):
            logger.info(f"Skipping {rank=} as it has already been completed.")
            return PipelineStats()  # todo: fetch the original stats file (?)
        add_task_logger(self.logging_dir, rank, local_rank)
        # pipe data from one step to the next
        pipelined_data = None
        for pipeline_step in self.pipeline:
            if callable(pipeline_step):
                pipelined_data = pipeline_step(pipelined_data, rank, self.world_size)
            elif isinstance(pipeline_step, Sequence) and not isinstance(pipeline_step, str):
                pipelined_data = pipeline_step
            else:
                raise ValueError
        if pipelined_data:
            deque(pipelined_data, maxlen=0)
        logger.info(f"Processing done for {rank=}")
        stats = PipelineStats(
            [pipeline_step.stats for pipeline_step in self.pipeline if isinstance(pipeline_step, PipelineStep)]
        )
        stats.save_to_disk(self.logging_dir.open(f"stats/{rank:05d}.json"))
        self.mark_rank_as_completed(rank)
        return stats

    def is_rank_completed(self, rank: int):
        return self.logging_dir.to_input_folder().file_exists(f"completions/{rank:05d}")

    def mark_rank_as_completed(self, rank: int):
        self.logging_dir.open(f"completions/{rank:05d}").close()
