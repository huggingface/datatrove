import dataclasses
import os.path
from abc import ABC, abstractmethod
from collections import Counter
from string import Template
from types import MethodType
from typing import IO, Callable

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


class DiskWriter(PipelineStep, ABC):
    """
        Base writer block to save data to disk.

    Args:
        output_folder: a str, tuple or DataFolder where data should be saved
        output_filename: the filename to use when saving data, including extension. Can contain placeholders such as `${rank}` or metadata tags `${tag}`
        compression: if any compression scheme should be used. By default, "infer" - will be guessed from the filename
        adapter: a custom function to "adapt" the Document format to the desired output format
    """

    default_output_filename: str = None
    type = "💽 - WRITER"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: str | None = "infer",
        adapter: Callable = None,
        mode: str = "wt",
        expand_metadata: bool = False,
        max_file_size: int = -1,  # in bytes. -1 for unlimited
    ):
        super().__init__()
        self.compression = compression
        self.output_folder = get_datafolder(output_folder)
        output_filename = output_filename or self.default_output_filename
        if self.compression == "gzip" and not output_filename.endswith(".gz"):
            output_filename += ".gz"
        self.max_file_size = max_file_size
        self.file_id_counter = Counter()
        if self.max_file_size > 0 and mode != "wb":
            raise ValueError("Can only specify `max_file_size` when writing in binary mode!")
        self.output_filename = Template(output_filename)
        self.output_mg = self.output_folder.get_output_file_manager(mode=mode, compression=compression)
        self.adapter = MethodType(adapter, self) if adapter else self._default_adapter
        self.expand_metadata = expand_metadata

    def _default_adapter(self, document: Document) -> dict:
        """
        You can create your own adapter that returns a dictionary in your preferred format
        Args:
            document: document to format

        Returns: a dictionary to write to disk

        """
        data = {key: val for key, val in dataclasses.asdict(document).items() if val}
        if self.expand_metadata and "metadata" in data:
            data |= data.pop("metadata")
        return data

    def __enter__(self):
        return self

    def close(self):
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

    def _on_file_switch(self, _original_name, old_filename, _new_filename):
        """
            Called when we are switching file from "old_filename" to "new_filename" (original_name is the filename
            without 000_, 001_, etc)
        Args:
            _original_name: name without file counter
            old_filename: old full filename
            _new_filename: new full filename

        """
        self.output_mg.pop(old_filename).close()

    def _get_filename_with_file_id(self, filename):
        """
            Prepend a file id to the base filename for when we are splitting files at a given max size
        Args:
            filename: filename without file id

        Returns: formatted filename

        """
        return f"{os.path.dirname(filename)}/{self.file_id_counter[filename]:03d}_{os.path.basename(filename)}"

    def write(self, document: Document, rank: int = 0, **kwargs):
        """
        Top level method to write a `Document` to disk. Will compute its output filename, adapt it to desired output format, write it and save stats.
        Args:
            document:
            rank:
            **kwargs: for the filename

        Returns:

        """
        original_name = output_filename = self._get_output_filename(document, rank, **kwargs)
        # we possibly have to change file
        if self.max_file_size > 0:
            # get size of current file
            output_filename = self._get_filename_with_file_id(original_name)
            # we have to switch file!
            if self.output_mg.get_file(output_filename).tell() >= self.max_file_size:
                self.file_id_counter[original_name] += 1
                new_output_filename = self._get_filename_with_file_id(original_name)
                self._on_file_switch(original_name, output_filename, new_output_filename)
                output_filename = new_output_filename
        # actually write
        self._write(self.adapter(document), self.output_mg.get_file(output_filename), original_name)
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
