import heapq
import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, TopKConfig
from datatrove.utils.stats import MetricStats, MetricStatsDict


STATS_MERGED_NAME = "metric.json"


class StatsMerger(PipelineStep):
    """
    Datatrove block for merging partial stats files into a single file.
    Each stat is of type MetricStatsDict saved in output_folder/{group}/{stat_name}/metric.json
    Args:
        input_folder: The folder used for saving stats files of SummaryStats block.
        output_folder: The folder where the merged stats will be saved.
        remove_input: Whether to remove the input files after merging.
        top_k: The configuration for compressing the statistics.
            Each group in top_k_groups will truncate the statistics to the top k keys.
    """

    type = "📊 - STATS"
    name = "🔗 Merging stats"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        remove_input: bool = False,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.remove_input = remove_input
        self.top_k_config = top_k_config

    def get_leaf_non_empty_folders(self):
        return sorted([path for path, folders, files in self.input_folder.walk("") if not folders and files])

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        Args:
          data: DocumentsPipeline:  (Default value = None)
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Each node will read a folder with stats files and merge them into a single file
        """
        folders_shard = self.get_leaf_non_empty_folders()[rank::world_size]
        logger.info(f"Merging {len(folders_shard)} stat folders")
        with self.track_time():
            for folder in tqdm(folders_shard):
                input_files = self.input_folder.glob(f"{folder}/[0-9][0-9][0-9][0-9][0-9].json")
                logger.info(f"Processing folder {folder} with {len(input_files)} files")

                stat = MetricStatsDict()
                for file in tqdm(input_files):
                    # Use inplace add to avoid creating a new dict
                    with self.input_folder.open(file, "rt") as f:
                        for key, item in json.load(f).items():
                            stat[key] += MetricStats.from_dict(item)

                with self.output_folder.open(f"{folder}/{STATS_MERGED_NAME}", "wt") as f:
                    group_name = Path(folder).parent.name
                    if group_name in self.top_k_config.top_k_groups:
                        top_k_keys = heapq.nlargest(self.top_k_config.top_k, stat, key=lambda x: stat.get(x).n)
                        stat = MetricStatsDict(init={s: stat.get(s) for s in top_k_keys})
                    json.dump(stat.to_dict(), f)

                if self.remove_input:
                    for file in input_files:
                        self.input_folder.rm(file)

        if data:
            yield from data
