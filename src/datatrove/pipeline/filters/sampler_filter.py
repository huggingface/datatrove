from numpy.random import default_rng

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter


class SamplerFilter(BaseFilter):
    name = "🎲 Sampler"

    def __init__(
        self,
        rate: float | None = 0.5,  # rate to KEEP
        seed: int = None,
        **kwargs,
    ):
        """ """
        super().__init__(**kwargs)
        self.rate = rate
        self.uniform = default_rng(seed).uniform

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        return self.uniform() < self.rate
