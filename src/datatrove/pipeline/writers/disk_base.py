from abc import ABC, abstractmethod
from string import Template

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import BaseOutputDataFolder, OutputDataFile
from datatrove.pipeline.base import PipelineStep


class DiskWriter(PipelineStep, ABC):
    default_output_filename: str = None
    type = "💽 - WRITER"

    def __init__(self, output_folder: BaseOutputDataFolder, output_filename: str = None, **kwargs):
        super().__init__(**kwargs)
        self.output_folder = output_folder
        self.output_filename = Template(output_filename or self.default_output_filename)

    def __enter__(self):
        return self

    def close(self):
        self.output_folder.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_output_filename(self, document: Document, rank: int = 0):
        return self.output_filename.substitute(
            {"rank": str(rank).zfill(5), "data_id": document.data_id, **document.metadata}
        )

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    @abstractmethod
    def _write(self, document: Document, file_handler):
        raise NotImplementedError

    def open(self, output_filename):
        return self.output_folder.open(output_filename)

    def write(self, document: Document, rank: int = 0):
        output_filename = self._get_output_filename(document, rank)
        output_file: OutputDataFile = self.open(output_filename)
        self._write(document, output_file)
        output_file.nr_documents += 1

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self:
            for document in data:
                with self.stats.time_manager:
                    self.write(document, rank)
                yield document
