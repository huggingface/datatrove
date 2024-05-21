from abc import ABC, abstractmethod
from itertools import chain

from datatrove.data import Document, DocumentsPipeline
from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.stats import Stats


class PipelineStep(ABC):
    """Base pipeline block, all blocks should inherit from this one.
        Takes care of some general things such as handling dependencies, and stats

    Args:
        name: Name of the step
        type: Type of the step
            Types are high-level categories of steps, e.g. "Reader", "Tokenizer", "Filters", etc.
    """

    name: str = None
    type: str = None

    def __new__(cls, *args, **kwargs):
        """
            Checks if this block or its superclasses' dependencies are installed and raises an error otherwise.
        Args:
            *args:
            **kwargs:
        """
        required_dependencies = chain.from_iterable(getattr(t, "_requires_dependencies", []) for t in cls.mro())
        if required_dependencies:
            check_required_dependencies(cls.__name__, required_dependencies)
        return super().__new__(cls)

    def __init__(self):
        super().__init__()
        self.stats = Stats(str(self))

    def stat_update(self, *labels, value: int = 1, unit: str = None):
        """
        Register statistics. `stat_update("metric1", "metric2")` will add 1 to the count of both metrics. Using
        `stat_update("mymetric", value=15)` will increment the value of "mymetric" by 15, and 15 will be used to
        compute the mean, min, max, std dev, etc for "mymetric" You can also define the unit `stat_update("tokens",
        value=123, unit="batch")` will then display /batch as unit.

        Args:
          *labels: names of stats to change
          value: int:  (Default value = 1)
          unit: str:  (Default value = None) None is treated as doc (so when printing you will see /doc)

        Returns:

        """
        for label in labels:
            self.stats[label].update(value, unit)

    def update_doc_stats(self, document: Document):
        """
            Compute some general doc related statistics, such as length of each document in characters and also in
            tokens (if available)
        Args:
          document: Document:

        Returns:

        """
        self.stat_update("doc_len", value=len(document.text), unit="doc")
        if token_count := document.metadata.get("token_count", None):
            self.stat_update("doc_len_tokens", value=token_count, unit="doc")

    def track_time(self, unit: str = None):
        """
            Track the time a given block of code takes to run and add it to statistics. If this block is not applied
            on a document level, please specify "unit"
        Args:
          unit: str:  (Default value = None)

        Returns:

        """
        if unit:
            self.stats.time_stats.unit = unit
        return self.stats.time_stats

    def __repr__(self):
        return f"{self.type}: {self.name}"

    @abstractmethod
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        Main entrypoint for any pipeline step. `data` is a generator of `Document`, and this method should
        yield `Document` (either add new documents if it is reading them, modify their content or metadata,
        or drop a few if it is a filter)

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0) used when each worker needs to choose a shard of data to work on
          world_size: int:  (Default value = 1) used when each worker needs to choose a shard of data to work on

        Returns:

        """
        if data:
            yield from data

    def __call__(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
            Shorthand way of calling the `run` method.
            block = Block()
            for resultdoc in block():
                ...
        Args:
            data:
            rank:
            world_size:

        Returns:

        """
        return self.run(data, rank, world_size)
