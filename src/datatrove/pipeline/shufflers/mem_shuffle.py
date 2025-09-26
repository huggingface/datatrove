from concurrent.futures import ThreadPoolExecutor
from random import random
from datatrove.pipeline.readers import JsonlReader
import heapq
import itertools
from loguru import logger
from tqdm import tqdm
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.io import get_datafolder
from random import shuffle
from datatrove.io import DataFolderLike
from typing import Callable

from datatrove.pipeline.writers.jsonl import JsonlWriter

class Shuffled(PipelineStep):
    name = "🔀 Shuffle"
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        all_data = []

        with self.track_time():
            for d in data:
                all_data.append(d)
            shuffle(all_data)

        for d in all_data:
            yield d


class Shuffle(PipelineStep):
    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        num_out_files: int,
    ):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_out_files = num_out_files

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if rank != 0:
            logger.info(f"Shuffle rank {rank} is not the master rank, skipping")
            return

        df = get_datafolder(self.input_folder)
        with self.track_time():
            def load_file(file_path):
                jsonl_reader = JsonlReader(self.input_folder)
                return list(jsonl_reader.read_file(file_path))

            def write_file(args):
                batch, i = args
                writer = JsonlWriter(
                    output_folder=self.output_folder,
                )
                for record in batch:
                    writer.write(record, rank=i)
                writer.close()

            file_paths = df.list_files(glob_pattern="*.jsonl.gz")
            with ThreadPoolExecutor() as executor:
                file_records = []
                results = tqdm(executor.map(load_file, file_paths), desc="Loading files")
                for records in results:
                    file_records.append(records)

            file_records = list(itertools.chain(*file_records))
            random.shuffle(file_records)

            with ThreadPoolExecutor() as executor:
                list(tqdm(executor.map(write_file, [(batch, i) for i, batch in enumerate(file_records)]), desc="Writing files"))

        yield from data