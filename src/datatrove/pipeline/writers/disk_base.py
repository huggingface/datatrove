import dataclasses
import os.path
import random
import time
from abc import ABC, abstractmethod
from collections import Counter
from string import Template
from types import MethodType
from typing import IO, Callable

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import StatHints


HF_RETRYABLE_MESSAGES = (
    "A commit has happened since",
    "Precondition Failed",
    "Too Many Requests",
    "rate limit",
    "maximum queue size reached",
    "maximum time in concurrency queue reached",
    "lfs-verify",
    "MerkleDB Shard error",
)


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

    HF_MAX_RETRIES = 12
    HF_BASE_DELAY = 0.1

    @staticmethod
    def is_retryable_hf_hub_error(error_message: str) -> bool:
        """Return True for HF Hub errors we explicitly retry."""
        return any(message in error_message for message in HF_RETRYABLE_MESSAGES)

    def _retry_hf_hub_operation(self, operation_name: str, target: str, operation: Callable[[], object]) -> object:
        """Retry a transient HF Hub operation with exponential backoff."""
        last_error: Exception | None = None
        for attempt in range(self.HF_MAX_RETRIES):
            try:
                return operation()
            except Exception as error:
                if not self.is_retryable_hf_hub_error(str(error)):
                    raise
                last_error = error
                if attempt == self.HF_MAX_RETRIES - 1:
                    break
                delay = self.HF_BASE_DELAY * 2**attempt + random.uniform(0, 2)
                logger.warning(
                    f"HF Hub {operation_name} error for {target}, retrying in {delay:.1f}s "
                    f"({attempt + 1}/{self.HF_MAX_RETRIES})..."
                )
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed HF Hub {operation_name} for {target} after retries.")

    def _get_output_file_with_retry(self, output_filename: str) -> IO:
        """Get or open an output file with retries on transient HF Hub failures.

        This is mainly used for HF-backed output folders where open calls can
        fail transiently under high API pressure (for example 429/412 responses).
        """
        return self._retry_hf_hub_operation(
            operation_name="open",
            target=output_filename,
            operation=lambda: self.output_mg.get_file(output_filename),
        )

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: str | None = "infer",
        adapter: Callable = None,
        mode: str = "wt",
        expand_metadata: bool = False,
        max_file_size: int = -1,  # in bytes. -1 for unlimited
        save_media_bytes: bool = False,
    ):
        super().__init__()
        self.compression = compression
        self.output_folder = get_datafolder(output_folder)
        output_filename = output_filename or self.default_output_filename
        if self.compression == "gzip" and not output_filename.endswith(".gz"):
            output_filename += ".gz"
        elif self.compression == "zstd" and not output_filename.endswith(".zst"):
            output_filename += ".zst"
        self.max_file_size = max_file_size
        self.file_id_counter = Counter()
        if self.max_file_size > 0 and mode != "wb":
            raise ValueError("Can only specify `max_file_size` when writing in binary mode!")

        self.output_filename = Template(output_filename)
        if self.output_filename.safe_substitute(rank=0) == output_filename:
            logger.warning(
                f"Output filename template '{output_filename}' does not include ${{rank}}; parallel workers may overwrite outputs."
            )
        self.output_mg = self.output_folder.get_output_file_manager(mode=mode, compression=compression)
        self.adapter = MethodType(adapter, self) if adapter else self._default_adapter
        self.expand_metadata = expand_metadata
        self.save_media_bytes = save_media_bytes

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

        if not self.save_media_bytes and "media" in data:
            data["media"] = [
                {
                    **media,
                    "media_bytes": None,
                }
                for media in data["media"]
            ]
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
        self.close_file(old_filename)

    def close_file(self, filename):
        """Close a file and remove it from the output manager.

        Retries on known HF Hub commit race/congestion errors with exponential
        backoff. On failure, fsspec marks the file closed but the temp file
        persists, so we retry the upload_file call directly.

        Args:
            filename: filename to close
        """
        if self.max_file_size > 0 and filename not in self.output_mg.get_open_files():
            filename = self._get_filename_with_file_id(filename)
        file_obj = self.output_mg.pop(filename)
        try:
            file_obj.close()
        except Exception as close_error:
            err_str = str(close_error)
            has_hf_upload_state = (
                hasattr(file_obj, "fs") and hasattr(file_obj, "temp_file") and hasattr(file_obj, "resolved_path")
            )
            if not self.is_retryable_hf_hub_error(err_str) or not has_hf_upload_state:
                raise
            # fsspec marks the file closed, but the temp file still exists on disk.
            # Retry the HF upload_file call directly with exponential backoff.
            api = file_obj.fs._api

            def _upload() -> None:
                api.upload_file(
                    path_or_fileobj=file_obj.temp_file.name,
                    path_in_repo=file_obj.resolved_path.path_in_repo,
                    repo_id=file_obj.resolved_path.repo_id,
                    token=file_obj.fs.token,
                    repo_type=file_obj.resolved_path.repo_type,
                    revision=file_obj.resolved_path.revision,
                )

            self._retry_hf_hub_operation(
                operation_name="upload",
                target=filename,
                operation=_upload,
            )
            if os.path.exists(file_obj.temp_file.name):
                os.remove(file_obj.temp_file.name)

    def _get_filename_with_file_id(self, filename):
        """
            Prepend a file id to the base filename for when we are splitting files at a given max size
        Args:
            filename: filename without file id

        Returns: formatted filename

        """
        if os.path.dirname(filename):
            return f"{os.path.dirname(filename)}/{self.file_id_counter[filename]:03d}_{os.path.basename(filename)}"
        return f"{self.file_id_counter[filename]:03d}_{os.path.basename(filename)}"

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
            if self._get_output_file_with_retry(output_filename).tell() >= self.max_file_size:
                self.file_id_counter[original_name] += 1
                new_output_filename = self._get_filename_with_file_id(original_name)
                self._on_file_switch(original_name, output_filename, new_output_filename)
                output_filename = new_output_filename
        # actually write
        self._write(self.adapter(document), self._get_output_file_with_retry(output_filename), original_name)
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
