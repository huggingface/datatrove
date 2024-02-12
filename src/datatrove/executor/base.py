import dataclasses
import json
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from typing import Callable

from loguru import logger

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import add_task_logger, close_task_logger, get_random_str, get_timestamp, log_pipeline
from datatrove.utils.stats import PipelineStats


class PipelineExecutor(ABC):
    """ Base class for pipeline executors (local, slurm, etc.)

    Args:
        pipeline: a list of PipelineStep and/or custom functions
            with arguments (data: DocumentsPipeline, rank: int, world_size: int)
        logging_dir: where to save logs, stats, etc. Should be parsable into a datatrove.io.DataFolder
        skip_completed: whether to skip tasks that were completed in
                previous runs. default: True
    """
    @abstractmethod
    def __init__(
        self,
        pipeline: list[PipelineStep | Callable],
        logging_dir: DataFolderLike = None,
        skip_completed: bool = True,
    ):
        self.pipeline: list[PipelineStep | Callable] = pipeline
        self.logging_dir = get_datafolder(logging_dir if logging_dir else f"logs/{get_timestamp()}_{get_random_str()}")
        self.skip_completed = skip_completed

    @abstractmethod
    def run(self):
        """ Run the pipeline on all tasks.
        """
        pass

    @property
    @abstractmethod
    def world_size(self):
        """ Return the total number of tasks to run the pipeline on."""
        return 0

    def _run_for_rank(self, rank: int, local_rank: int = 0) -> PipelineStats:
        """ Run the pipeline for a single rank.

        Args:
            rank: the rank (in the world size) this pipeline worker is running on
            local_rank: the local rank (default: 0) – used for limiting logging verbosity
        """
        if self.is_rank_completed(rank):
            logger.info(f"Skipping {rank=} as it has already been completed.")
            return PipelineStats()
        logfile = add_task_logger(self.logging_dir, rank, 0)  # local_rank)
        log_pipeline(self.pipeline)
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

    def is_rank_completed(self, rank: int):
        """ Check if a rank (a task in the world size) is already completed.

        Args:
            rank (int): The rank to check
        """
        return self.skip_completed and self.logging_dir.isfile(f"completions/{rank:05d}")

    def mark_rank_as_completed(self, rank: int):
        """ Mark a rank (a task in the world size) as completed.
            We're using files in the logging_dir folder (at logging_dir/completions) to mark completion.

        Args:
            rank (int): The rank to mark as completed
        """
        self.logging_dir.open(f"completions/{rank:05d}", "w").close()

    def get_incomplete_ranks(self):
        """ Get the list of ranks that are not yet completed.
            This is based on the presence of files in the logging_dir/completions folder.
        
        Returns:
            list[int]: list of ranks that are not yet completed
        """
        completed = set(self.logging_dir.list_files("completions"))
        return list(
            filter(
                lambda rank: not self.skip_completed or f"completions/{rank:05d}" not in completed,
                range(self.world_size),
            )
        )

    def to_json(self, indent=4):
        """ Convert the executor to a JSON string.
        """
        data = self.__dict__
        data["pipeline"] = [{a: b for a, b in x.__dict__.items() if a != "stats"} for x in data["pipeline"]]
        return json.dumps(data, indent=indent)

    def save_executor_as_json(self, indent: int = 4):
        """ Save the executor as a JSON file in the logging directory.
        """
        with self.logging_dir.open("executor.json", "w") as f:
            json.dump(self, f, cls=ExecutorJSONEncoder, indent=indent)


class ExecutorJSONEncoder(json.JSONEncoder):
    """ Custom JSON encoder for the PipelineExecutor class
    """
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, PipelineExecutor):
            return o.__dict__ | {"world_size": o.world_size}
        if isinstance(o, PipelineStep):
            return {a: b for a, b in o.__dict__.items() if a != "stats"}
        return str(o)
