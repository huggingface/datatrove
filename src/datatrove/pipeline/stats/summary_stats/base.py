import heapq
import json
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Literal, get_args

from loguru import logger

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.stats import MetricStatsDict


GROUP = Literal["summary", "histogram", "fqdn", "suffix"]


@dataclass
class TopKConfig:
    """
    Configuration for compressing the statistics.
    Each group in top_k_groups will truncate the statistics to the top k keys.
    This lowers memory usage and speeds up the merging in second-stage.

    If run in distributed mode, each node will create its own top_k_keys, which
    leads to inconsistent top_k_keys between nodes. To acount for this, set around
    0.8*top_k as the number of top_k_keys for merging step.
    """

    top_k_groups: List[GROUP]
    top_k: int


DEFAULT_TOP_K_CONFIG = TopKConfig(top_k_groups=["fqdn", "suffix"], top_k=100_000)

STAT_TYPE = int | float


class BaseStats(PipelineStep):
    """
    Datatrove block for computing statistics of dataset.
    Each stat is of type MetricStatsDict saved in output_folder/{group}/{stat_name}/{rank:05d}.json
    Args:
        output_folder: The folder where the statistics will be saved.
        groups_to_compute: The groups of statistics to compute.
        histogram_round_digits: The number of digits to round the histogram values to.
            This ensures reasonable number of bins.
        top_k_config: The configuration for compressing the statistics.
            Each group in top_k_groups will truncate the statistics to the top k keys.
            This lowers memory usage and speeds up the merging in second-stage.
    """

    type = "📊 - STATS"
    name = "👑 Summary stats"
    _requires_dependencies = ["tldextract"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        histogram_round_digits: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        from tldextract import TLDExtract

        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.groups = groups_to_compute
        self.histogram_round_digits = histogram_round_digits
        self.top_k_cfg = top_k_config
        self.tld_extractor = TLDExtract()

    @abstractmethod
    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        """
        Abstract method for extracting stats from a document.
        Args:
            doc: The document to extract stats from.

        Returns:
            A dictionary of statistics, where the key is the stat name and the value is the stat value.
        """
        raise NotImplementedError()

    def get_kv(self, doc: Document, value: STAT_TYPE, group_name: GROUP) -> tuple[str, STAT_TYPE]:
        if group_name == "histogram":
            # Use rounding to reduce then number of values for histogram
            return str(round(value, self.histogram_round_digits)), 1
        elif group_name == "summary":
            return "summary", value
        elif group_name == "fqdn":
            return self.tld_extractor.extract_str(doc.metadata["url"]).fqdn, value
        elif group_name == "suffix":
            return self.tld_extractor.extract_str(doc.metadata["url"]).suffix, value
        else:
            raise ValueError(f"Unknown group name: {group_name}")

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        groups_dicts = {group: defaultdict(MetricStatsDict) for group in self.groups}
        with self.track_time():
            for doc in data:
                try:
                    doc_stats = self.extract_stats(doc)
                except Exception as e:
                    logger.error(f"Error while extracting stats from document {doc.id}", exc_info=e)
                    raise e

                for group, counters in groups_dicts.items():
                    for stat, value in doc_stats.items():
                        key, value = self.get_kv(doc, value, group)
                        counters[stat][key] += value

                yield doc

        # save to disk
        for group, stats_dict in groups_dicts.items():
            group_top_k_keys = None

            for stat_name, stat_values in stats_dict.items():
                if group in self.top_k_cfg.top_k_groups:
                    # We don't have to compure this for every stat in group, as stat.n will be constant
                    if group_top_k_keys is None:
                        group_top_k_keys = heapq.nlargest(
                            self.top_k_cfg.top_k, stat_values, key=lambda x: stat_values.get(x).n
                        )

                    stat_values = MetricStatsDict(init={s: stat_values.get(s) for s in group_top_k_keys})

                with self.output_folder.open(f"{group}/{stat_name}/{rank:05d}.json", "wt") as f:
                    json.dump(stat_values.to_dict(), f)
