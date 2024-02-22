import dataclasses
from abc import ABC, abstractmethod
from string import Template
from typing import IO, Callable

from loguru import logger

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


def _default_adapter(document: Document) -> dict:
    """
    You can create your own adapter that returns a dictionary in your preferred format
    Args:
        document: document to format

    Returns: a dictionary to write to disk

    """
    return {key: val for key, val in dataclasses.asdict(document).items() if val}


class DiskWriter(PipelineStep, ABC):
    default_output_filename: str = None
    type = "💽 - WRITER"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: str | None = "infer",
        adapter: Callable = None,
        mode: str = "wt",
    ):
        """
            Base writer block to save data to disk.
        Args:
            output_folder: a str, tuple or DataFolder where data should be saved
            output_filename: the filename to use when saving data, including extension. Can contain placeholders such as `${rank}` or metadata tags `${tag}`
            compression: if any compression scheme should be used. By default, "infer" - will be guessed from the filename
            adapter: a custom function to "adapt" the Document format to the desired output format
        """
        super().__init__()
        self.compression = compression
        self.output_folder = get_datafolder(output_folder)
        output_filename = output_filename or self.default_output_filename
        if self.compression == "gzip" and not output_filename.endswith(".gz"):
            output_filename += ".gz"
        self.output_filename = Template(output_filename)
        self.output_mg = self.output_folder.get_output_file_manager(mode=mode, compression=compression)
        self.adapter = adapter if adapter else _default_adapter

    def __enter__(self):
        logger.info("RUNNING ENTER")
        return self

    def close(self):
        logger.info("CLOSE ON MAIN")
        self.output_mg.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_output_filename(self, document: Document, rank: int | str = 0, **kwargs) -> str:
        """
            Get the output path for a given document, based on any possible tag replacement.
            Example filename with `rank` tag: "${rank}.jsonl.gz"
            Tags replaced:
            - `rank`: rank of the current worker. Important as to avoid multiple workers writing to the same file
            - `id`: the document's id
            - metadata: any metadata field can be replaced directly
            - kwargs: any additional kwargs passed to this function
        Args:
            document: the document for which the output path should be determined
            rank: the rank of the current worker
            **kwargs: any additional tags to replace in the filename

        Returns: the final replaced path for this document

        """
        return self.output_filename.substitute(
            {"rank": str(rank).zfill(5), "id": document.id, **document.metadata, **kwargs}
        )

    @abstractmethod
    def _write(self, document: dict, file_handler: IO, filename: str):
        """
        Main method that subclasses should implement. Receives an adapted (after applying self.adapter) dictionary with data to save to `file_handler`
        Args:
            document: dictionary with the data to save
            file_handler: file_handler where it should be saved
            filename: to use as a key for writer helpers and other data
        Returns:

        """
        raise NotImplementedError

    def write(self, document: Document, rank: int = 0, **kwargs):
        """
        Top level method to write a `Document` to disk. Will compute its output filename, adapt it to desired output format, write it and save stats.
        Args:
            document:
            rank:
            **kwargs: for the filename

        Returns:

        """
        output_filename = self._get_output_filename(document, rank, **kwargs)
        self._write(self.adapter(document), self.output_mg.get_file(output_filename), output_filename)
        self.stat_update(self._get_output_filename(document, "XXXXX", **kwargs))
        self.stat_update(StatHints.total)
        self.update_doc_stats(document)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        Simply call `write` for each document
        Args:
            data:
            rank:
            world_size:

        Returns:

        """
        with self:
            for document in data:
                with self.track_time():
                    self.write(document, rank)
                yield document
