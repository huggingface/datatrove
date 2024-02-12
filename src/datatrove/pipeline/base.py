from abc import ABC, abstractmethod
from itertools import chain
from typing import NoReturn

from datatrove.data import Document, DocumentsPipeline
from datatrove.utils._import_utils import _is_package_available
from datatrove.utils.stats import Stats


class PipelineStep(ABC):
    """ Base class for all pipeline steps.

    Args:
        name: Name of the step
        type: Type of the step
            Types are high-level categories of steps, e.g. "Reader", "Tokenizer", "Filters", etc.
    """
    name: str = None
    type: str = None

    def __new__(cls, *args, **kwargs):
        """ Mostly to check for required dependencies before creating the instance.
        """
        required_dependencies = chain.from_iterable(getattr(t, "_requires_dependencies", []) for t in cls.mro())
        if required_dependencies:
            missing_dependencies: dict[str, str] = {}
            for dependency in required_dependencies:
                dependency = dependency if isinstance(dependency, tuple) else (dependency, dependency)
                package_name, pip_name = dependency
                if not _is_package_available(package_name):
                    missing_dependencies[package_name] = pip_name
            if missing_dependencies:
                _raise_error_for_missing_dependencies(cls.__name__, missing_dependencies)
        return super().__new__(cls)

    def __init__(self):
        self.stats = Stats(str(self))

    def stat_update(self, *labels, value: int = 1, unit: str = None):
        """ Update the stats for the step with the given labels and value

        Args:
            labels (positional args): a list of labels to update
            value: int
                The value to update
            unit: str
                The unit of the value
        """
        for label in labels:
            self.stats[label].update(value, unit)

    def update_doc_stats(self, document: Document):
        """ Update the stats to add the document length (in characters and tokens if available) 
        """
        self.stats["doc_len"] += len(document.text)
        if token_count := document.metadata.get("token_count", None):
            self.stats["doc_len_tokens"] += token_count

    def track_time(self, unit: str = None):
        """ Helper to track time stats
        """
        if unit:
            self.stats.time_stats.unit = unit
        return self.stats.time_stats

    def __repr__(self):
        return f"{self.type}: {self.name}"

    @abstractmethod
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """ Main method to run the step. Main entry point to a class instance when called (__call__)
        
        Args:
            data: DocumentsPipeline
                The data to be processed as a Generator typically created by a Reader initial pipeline step
            rank: int
                The rank of the process
            world_size: int
                The total number of processes
        """
        if data:
            yield from data

    def __call__(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        return self.run(data, rank, world_size)


def _raise_error_for_missing_dependencies(step_name: str, dependencies: dict[str, str]) -> NoReturn:
    """ Helper to raise an ImportError for missing dependencies

    Args:
        step_name: str
            The name of the step
        dependencies: dict[str, str]
            The missing dependencies
    """
    dependencies = dict(sorted(dependencies.items()))
    package_names = list(dependencies)
    if len(dependencies) > 1:
        package_names = (
            f"{','.join('`' + package_name + '`' for package_name in package_names[:-1])} and `{package_names[-1]}`"
        )
    else:
        package_names = f"`{package_names[0]}`"
    raise ImportError(
        f"Please install {package_names} to use {step_name} (`pip install {' '.join(list(dependencies.values()))}`)."
    )
