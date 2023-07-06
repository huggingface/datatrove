from abc import abstractmethod

from loguru import logger

from datatrove.data import DocumentsPipeline
from datatrove.io import BaseInputDataFolder, InputDataFile
from datatrove.pipeline.base import PipelineStep


class BaseReader(PipelineStep):
    type = "📖 - READER"

    def __init__(self, data_folder: BaseInputDataFolder, **kwargs):
        super().__init__(**kwargs)
        self.data_folder = data_folder

    @abstractmethod
    def read_file(self, datafile: InputDataFile):
        raise NotImplementedError

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.data_folder.set_lock(dl_lock)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data
        for datafile in self.data_folder.get_files_shard(rank, world_size):
            logger.info(f"Reading input file {datafile.path}")
            yield from self.read_file(datafile)
