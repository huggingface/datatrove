from abc import abstractmethod
from contextlib import nullcontext

from loguru import logger
from tqdm import tqdm

from datatrove.data import DocumentsPipeline
from datatrove.io import BaseInputDataFolder, InputDataFile
from datatrove.pipeline.base import PipelineStep


class BaseReader(PipelineStep):
    type = "📖 - READER"

    def __init__(self, data_folder: BaseInputDataFolder, limit: int = -1, progress: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.data_folder = data_folder
        self.limit = limit
        self.progress = progress

    @abstractmethod
    def read_file(self, datafile: InputDataFile):
        raise NotImplementedError

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.data_folder.set_lock(dl_lock)

    def read_files_shard(self, shard):
        li = 0
        with tqdm(total=self.limit if self.limit != -1 else None) if self.progress else nullcontext() as pbar:
            for datafile in shard:
                logger.info(f"Reading input file {datafile.path}")
                for document in self.read_file(datafile):
                    if self.limit != -1 and li >= self.limit:
                        return
                    yield document
                    if pbar:
                        pbar.update()
                    li += 1

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data
        yield from self.read_files_shard(self.data_folder.get_files_shard(rank, world_size))
