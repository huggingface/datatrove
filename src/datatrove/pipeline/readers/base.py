from abc import abstractmethod

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.io import InputDataFolder, InputDataFile


class BaseReader(PipelineStep):

    def __init__(
            self,
            data_folder: InputDataFolder,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.data_folder = data_folder
        self.type = "📖 - READER"

    @abstractmethod
    def read_file(self, datafile: InputDataFile):
        raise NotImplementedError

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.data_folder.set_lock(dl_lock)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data
        for datafile in self.data_folder.get_files_shard(rank, world_size):
            yield from self.read_file(datafile)
