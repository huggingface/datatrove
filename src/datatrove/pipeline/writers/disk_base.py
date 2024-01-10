from abc import ABC, abstractmethod
from string import Template

from datatrove.data import Document, DocumentsPipeline
from datatrove.datafolder import ParsableDataFolder, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


class DiskWriter(PipelineStep, ABC):
    default_output_filename: str = None
    type = "💽 - WRITER"

    def __init__(
        self, output_folder: ParsableDataFolder, output_filename: str = None, compression: str | None = "infer"
    ):
        super().__init__()
        self.compression = compression
        self.output_folder = get_datafolder(output_folder)
        output_filename = output_filename or self.default_output_filename
        if self.compression == "gzip" and not output_filename.endswith(".gz"):
            output_filename += ".gz"
        self.output_filename = Template(output_filename)
        self.output_mg = self.output_folder.get_outputfile_manager(mode="wt", compression=compression)

    def __enter__(self):
        return self

    def close(self):
        self.output_mg.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_output_filename(self, document: Document, rank: int | str = 0, **kwargs):
        return self.output_filename.substitute(
            {"rank": str(rank).zfill(5), "data_id": document.data_id, **document.metadata, **kwargs}
        )

    @abstractmethod
    def _write(self, document: Document, file_handler):
        raise NotImplementedError

    def write(self, document: Document, rank: int = 0, **kwargs):
        output_filename = self._get_output_filename(document, rank, **kwargs)
        self._write(document, self.output_mg.get_file(output_filename))
        self.stat_update(self._get_output_filename(document, "XXXXX", **kwargs))
        self.stat_update(StatHints.total)
        self.update_doc_stats(document)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self:
            for document in data:
                with self.track_time():
                    self.write(document, rank)
                yield document
