from datatrove.data import DocumentsPipeline
from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep
from loguru import logger
from datatrove.pipeline.writers import JsonlWriter
from typing import Callable

from datatrove.io import DataFolderLike


class SortMemHeavy(PipelineStep):
    name = "SortMemHeavy"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        num_out_files: int,
        key: Callable,
        load_proc: int = 30,
        prefetch_size: int = 5000,
    ):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_out_files = num_out_files
        self.key = key
        self.load_proc = load_proc
        self.prefetch_size = prefetch_size
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        from concurrent.futures import ThreadPoolExecutor
        from datatrove.pipeline.readers import JsonlReader
        import collections
        import heapq
        import itertools
        from tqdm import tqdm

        df = get_datafolder(self.input_folder)


        class BufferedReader:
            def __init__(self, reader: JsonlReader, prefetch_size: int = 100):
                self.reader = reader
                self.prefetch_size = prefetch_size

            def get_length(self, path: str):
                return sum(1 for _ in self.reader.read_file(path))

            def open_file(self, path: str):
                return self._Iterator(self, path)

            class _Iterator:
                def __init__(self, bf_reader: "BufferedReader", path: str):
                    self.bf_reader = bf_reader
                    self.path = path
                    self.queue = collections.deque()
                    self.last_key = None
                    self.doc_iterable = self.bf_reader.reader.read_file(self.path)
                    self.prefetch()

                def prefetch(self):
                    while len(self.queue) < self.bf_reader.prefetch_size:
                        try:
                            self.queue.append(next(self.doc_iterable))
                        except StopIteration:
                            break

                def __iter__(self):
                    return self

                def __next__(self):
                    if self.queue:
                        return self.queue.popleft()

                    self.prefetch()
                    # If there is nothing left this will raise StopIteration
                    if not self.queue:
                        raise StopIteration
                    return self.queue.popleft()

        def ensure_sorted(iterable, key, path):
            last_key = None
            for record in iterable:
                current_key = key(record)
                if last_key and current_key < last_key:
                    raise ValueError(f"Records not sorted in file {path}")
                last_key = current_key
                yield record

        with self.track_time():
            paths = df.list_files(glob_pattern="*.jsonl.gz")
            readers = [BufferedReader(JsonlReader(df), prefetch_size=1000) for _ in paths]
            with ThreadPoolExecutor(max_workers=self.load_proc) as executor:
                # First we compute the length of each file
                total_records = sum(tqdm(executor.map(lambda args: args[0].get_length(args[1]), zip(readers, paths)), desc="Computing the length of files", total=len(readers)))
                if total_records == 0:
                    return data

                reader_iterables = list(tqdm(executor.map(lambda args: args[0].open_file(args[1]), zip(readers, paths)), desc="Prefetching files", total=len(readers)))

            merged_iterator = heapq.merge(*[ensure_sorted(reader_iterable, self.key, path) for reader_iterable, path in zip(reader_iterables, paths)], key=self.key)
            num_out_files = self.num_out_files
            if total_records < num_out_files:
                num_out_files = total_records
                logger.warning(f"Total records {total_records} are less than the number of output files {num_out_files}, using {num_out_files} files")

            batch_size = total_records // num_out_files

            # Prepare tqdm
            pbar = tqdm(desc="Writing records", total=total_records)

            for (i, batch) in enumerate(itertools.batched(merged_iterator, batch_size)):
                writer = JsonlWriter(
                    output_folder=self.output_folder,
                )

                file_batch_size = batch_size if i < num_out_files - 1 else total_records - batch_size * (num_out_files - 1)
                with writer:
                    for record in batch:
                        writer.write(record, rank=i)
                        pbar.update(1)
            pbar.close()
        return data

    


class Sorted(PipelineStep):
    name = "🔀 Sorted"
    def __init__(self, key: Callable):
        super().__init__()
        self.key = key

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        all_data = []

        with self.track_time():
            for d in data:
                all_data.append(d)
            all_data.sort(key=self.key)

        for d in all_data:
            yield d
