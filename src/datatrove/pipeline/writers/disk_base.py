from abc import abstractmethod, ABC
from collections.abc import Callable
from string import Template

from datatrove.data import DocumentsPipeline, Document
from datatrove.pipeline.base import PipelineStep
from datatrove.io import OutputDataFolder, OutputDataFile


class DiskWriter(PipelineStep, ABC):
    open_fn: Callable = None
    close_fn: Callable = None
    default_output_filename: str = None

    def __init__(
            self,
            output_folder: OutputDataFolder,
            output_filename: str = None,
            **kwargs
    ):
        self.output_folder = output_folder
        self.output_filename = Template(output_filename or self.default_output_filename)
        super().__init__(**kwargs)

    def __enter__(self):
        return self

    def close(self):
        self.output_folder.close(self.close_fn)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_output_filename(self, document: Document, rank: int = 0):
        return self.output_filename.substitute({
            'rank': str(rank).zfill(5),
            'data_id': document.data_id,
            **document.metadata
        })

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    @abstractmethod
    def _write(self, document: Document, file_handler):
        raise NotImplementedError

    def write(self, document: Document, rank: int = 0):
        output_filename = self._get_output_filename(document, rank)
        output_file: OutputDataFile = self.output_folder.get_file(output_filename, open_fn=self.open_fn)
        self._write(document, output_file)
        output_file.nr_documents += 1

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self:
            for document in data:
                self.write(document, rank)
