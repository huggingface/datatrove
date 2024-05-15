from dataclasses import dataclass
from typing import Literal


GROUP = Literal["summary", "histogram", "fqdn", "suffix"]


@dataclass(frozen=True)
class TopKConfig:
    """
    Configuration for compressing the statistics.
    Each group in top_k_groups will be truncated to the top k keys.
    This lowers memory usage and speeds up the merging in second-stage.

    If run in distributed mode, each node will create its own top_k_keys, which
    leads to inconsistent top_k_keys between nodes. To account for this, set around
    0.8*top_k as the number of top_k_keys for merging step.
    """

    top_k_groups: list[Literal["fqdn", "suffix"]]
    top_k: int


DEFAULT_TOP_K_CONFIG = TopKConfig(top_k_groups=["fqdn", "suffix"], top_k=100_000)

STAT_TYPE = int | float
