from typing import Optional, Tuple, Union

from numpy.random import default_rng

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class SamplerFilter(BaseFilter):
    name = "🎲 Sampler"

    def __init__(
        self, rate: Optional[float] = 0.5, seed: int = None, exclusion_writer: DiskWriter = None  # rate to KEEP
    ):
        """ """
        super().__init__(exclusion_writer)
        self.rate = rate
        self.uniform = default_rng(seed).uniform

    def filter(self, doc: Document) -> Union[bool, Tuple[bool, str]]:
        return self.uniform() < self.rate
