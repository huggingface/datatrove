import dataclasses
import json
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from typing import Callable

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import (
    add_task_logger,
    close_task_logger,
    get_random_str,
    get_timestamp,
    log_pipeline,
    logger,
)
from datatrove.utils.stats import PipelineStats


class PipelineExecutor(ABC):
    """Base class for pipeline executors (local, slurm, etc.)

    Args:
        pipeline: a list of PipelineStep and/or custom functions
            with arguments (data: DocumentsPipeline, rank: int, world_size: int)
        logging_dir: where to save logs, stats, etc. Should be parsable into a datatrove.io.DataFolder
        skip_completed: whether to skip tasks that were completed in
                previous runs. default: True
        randomize_start_duration: the maximum number of seconds to delay the start of each task.
    """

    @abstractmethod
    def __init__(
        self,
        pipeline: list[PipelineStep | Callable],
        logging_dir: DataFolderLike = None,
        skip_completed: bool = True,
        randomize_start_duration: int = 0,
    ):
        self.pipeline: list[PipelineStep | Callable] = pipeline
        self.logging_dir = get_datafolder(logging_dir if logging_dir else f"logs/{get_timestamp()}_{get_random_str()}")
        self.skip_completed = skip_completed
        self.randomize_start_duration = randomize_start_duration

    @abstractmethod
    def run(self):
        """Run the pipeline on all tasks.
        This method is responsible for correctly invoking `self._run_for_rank` for each task that is to be run.
        See slurm and local executor for example usage.
        """
        pass

    @property
    @abstractmethod
    def world_size(self) -> int:
        """
        Returns: the total number of tasks to consider. This is used for sharding data files, for example

        """
        return 0

    def _run_for_rank(self, rank: int, local_rank: int = 0) -> PipelineStats:
        """
            Main executor's method. Sets up logging, pipes data from each pipeline step to the next, saves statistics
            and marks tasks as completed.
        Args:
            rank: the rank that we want to run the pipeline for
            local_rank: at the moment this is only used for logging.
            Any task with local_rank != 0 will not print logs to console.

        Returns: the stats for this task

        """
        if self.is_rank_completed(rank):
            logger.info(f"Skipping {rank=} as it has already been completed.")
            return PipelineStats()
        logfile = add_task_logger(self.logging_dir, rank, local_rank)
        log_pipeline(self.pipeline)

        if self.randomize_start_duration > 0:
            time.sleep(random.randint(0, self.randomize_start_duration))
        try:
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

            logger.success(f"Processing done for {rank=}")

            # stats
            stats = PipelineStats(self.pipeline)
            with self.logging_dir.open(f"stats/{rank:05d}.json", "w") as f:
                stats.save_to_disk(f)
            logger.info(stats.get_repr(f"Task {rank}"))
            # completed
            self.mark_rank_as_completed(rank)
        except Exception as e:
            logger.exception(e)
            raise e
        finally:
            close_task_logger(logfile)
        return stats

    def is_rank_completed(self, rank: int) -> bool:
        """
            Checks if a given task has already been completed.
        Args:
            rank: the rank of the task to check

        Returns: whether task is already completed. If `skip_completed=False`, will always return `False`.

        """
        return self.skip_completed and self.logging_dir.isfile(f"completions/{rank:05d}")

    def mark_rank_as_completed(self, rank: int):
        """
            Marks a given task as completed.
            In practice this involves creating an empty file with the rank in the filename.
        Args:
            rank: the rank of the task to mark as completed

        Returns:

        """
        self.logging_dir.open(f"completions/{rank:05d}", "w").close()

    def get_incomplete_ranks(self, ranks=None) -> list[int]:
        """
            Gets a full list of ranks that are still incomplete.
            Usually faster than calling `is_rank_completed` for each task.
        Returns: list of ranks that are incomplete

        """
        completed = set(self.logging_dir.list_files("completions"))
        return list(
            filter(
                lambda rank: not self.skip_completed or f"completions/{rank:05d}" not in completed,
                ranks if ranks is not None else range(self.world_size),
            )
        )

    def to_json(self, indent=4) -> str:
        """
            Returns a json representation of this executor.
        Args:
            indent: how many spaces to use per indent

        Returns: json string

        """
        data = self.__dict__
        data["pipeline"] = [{a: b for a, b in x.__dict__.items() if a != "stats"} for x in data["pipeline"]]
        return json.dumps(data, indent=indent)

    def save_executor_as_json(self, indent: int = 4):
        """
            Save a json representation of this executor to a filesystem.
        Args:
            indent: how many spaces to use per indent

        Returns:

        """
        with self.logging_dir.open("executor.json", "w") as f:
            json.dump(self, f, cls=ExecutorJSONEncoder, indent=indent)


class ExecutorJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for the PipelineExecutor class"""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, PipelineExecutor):
            return o.__dict__ | {"world_size": o.world_size}
        if isinstance(o, PipelineStep):
            return {a: b for a, b in o.__dict__.items() if a != "stats"}
        return str(o)
