import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import IO

import zstandard as zstd

from datatrove.data import Document, DocumentsPipeline, Media
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from loguru import logger


class BinaryZstdReaderFast(PipelineStep):
    name = "📒⚡ - Binary Zstd Media Reader (Fast/Threaded)"
    type = "Media Reader"

    def __init__(
        self,
        data_folder: DataFolderLike,
        workers: int = 1,
        offset_byte_size: int = 4,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.workers = workers
        self.offset_byte_size = offset_byte_size

    def _init_thread_local_storage(self):
        """Initializes file pointer and decompressor state for the current thread's local storage."""
        # This method sets attributes on self.thread_local for the *current* thread
        self.thread_local.current_filename = None
        self.thread_local.current_fp = None  # Raw file pointer
        self.thread_local.decompressor = zstd.ZstdDecompressor()

    def _close_thread_local_fp(self):
        """Closes the file pointer stored in the current thread's local storage."""
        if hasattr(self.thread_local, "current_fp") and self.thread_local.current_fp:
            try:
                self.thread_local.current_fp.close()
            except Exception as e:
                logger.warning(
                    f"Error closing thread-local raw fp for {getattr(self.thread_local, 'current_filename', 'N/A')}: {e}"
                )
        # Reset state after closing
        if hasattr(self.thread_local, "current_filename"): # Check before trying to set
            self.thread_local.current_filename = None
            self.thread_local.current_fp = None

    def read_media_record(self, document: Document, media: Media) -> Document:
        """
        Reads a single zstd compressed media record using thread-local file pointers.
        Updates the media object in place. The document is returned for consistency
        but modification is in-place.
        """
        file_path = media.path
        record_start_offset = media.offset # Offset to the start of [length_bytes][compressed_data]

        if file_path is None or record_start_offset is None:
            logger.warning(f"Media {media.id} in doc {document.id} missing path or offset, skipping.")
            media.metadata["read_error"] = "Missing path or offset"
            self.stat_update("media_error_missing_info", value=1, unit="media")
            return document

        # Initialize thread-local storage for this worker if it's the first time.
        if not hasattr(self.thread_local, "current_fp"): # A sentinel attribute
            self._init_thread_local_storage()

        fp = self.thread_local.current_fp

        try:
            if not (
                self.thread_local.current_filename == file_path
                and fp
                and not fp.closed
            ):
                self._close_thread_local_fp() # Close previous if any
                # Open the raw file (no transparent compression by DataFolder)
                fp = self.data_folder.open(file_path, "rb", compression=None)
                self.thread_local.current_fp = fp
                self.thread_local.current_filename = file_path

            # Seek on the raw file pointer to the start of the record
            if fp.tell() != record_start_offset:
                fp.seek(record_start_offset)

            # Read the length of the compressed data
            len_bytes = fp.read(self.offset_byte_size)
            if len(len_bytes) != self.offset_byte_size:
                raise EOFError(
                    f"Could not read compressed length prefix ({len(len_bytes)}B/"
                    f"{self.offset_byte_size}B) at offset {record_start_offset} in {file_path}"
                )
            compressed_length = int.from_bytes(len_bytes, "big")

            if compressed_length == 0: # Handle empty media case
                 content_bytes = b""
            else:
                # Read the compressed data
                compressed_content_bytes = fp.read(compressed_length)
                if len(compressed_content_bytes) != compressed_length:
                    raise EOFError(
                        f"Could not read full compressed record ({len(compressed_content_bytes)}B/"
                        f"{compressed_length}B) after offset {record_start_offset + self.offset_byte_size} in {file_path}"
                    )
                # Decompress the data
                content_bytes = self.thread_local.decompressor.decompress(compressed_content_bytes)

            media.media_bytes = content_bytes
            self.stat_update("media_fetched", value=1, unit="media")
            self.stat_update("media_fetched_bytes", value=len(content_bytes), unit="bytes")

        except EOFError as e:
            logger.warning(f"EOFError reading media {media.id} from {file_path} at offset {record_start_offset}: {e}")
            media.metadata["read_error"] = f"EOFError: {e}"
            self.stat_update("media_error_eof", value=1, unit="media")
            self._close_thread_local_fp() # Close FP on EOF to avoid reuse of bad state
        except Exception as e:
            logger.error(f"Error reading media {media.id} from {file_path} at offset {record_start_offset}: {e}", exc_info=True)
            media.metadata["read_error"] = str(e)
            self.stat_update("media_error_exception", value=1, unit="media")
            self._close_thread_local_fp() # Close FP on other errors too

        return document # Document is modified in-place

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if data is None:
            yield from [] # Or simply return
            return

        self.thread_local = threading.local()
        self._init_thread_local_storage() # Initialize for the main thread

        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=f"{self.name}_worker") as executor:
            try:
                for document in data:
                    if document is None:
                        logger.warning("Received None document from upstream. Skipping.")
                        # Decide whether to yield None or just skip
                        # yield None
                        continue

                    media_futures = []
                    for media_item in document.media:
                        if media_item.path and media_item.offset is not None:
                            future = executor.submit(self.read_media_record, document, media_item)
                            media_futures.append(future)
                        else:
                            logger.warning(
                                f"Skipping media {media_item.id} in doc {document.id} (rank {rank}): "
                                f"missing path or offset."
                            )
                            media_item.metadata["read_error"] = "Missing path or offset"
                            self.stat_update("media_error_missing_info", value=1, unit="media")

                    # Wait for all media items of the current document to be processed
                    if media_futures: # Only wait if there were tasks
                        for f_idx, future_result in enumerate(executor.map(lambda f: f.result(), media_futures)):
                            # .result() will re-raise exceptions.
                            # read_media_record already handles logging and setting metadata.
                            # Here we just ensure all tasks complete.
                            # The document is modified in-place by read_media_record.
                            pass


                    yield document # Yield the document after all its media have been processed/attempted

            finally:
                logger.info(f"Finished processing all documents for {self.name} (rank {rank}). "
                            f"Shutting down executor and cleaning up main thread file pointer.")
                self._close_thread_local_fp() # Cleanup for the main thread