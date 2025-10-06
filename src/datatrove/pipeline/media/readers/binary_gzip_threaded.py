import gzip
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import IO

from datatrove.data import Document, DocumentsPipeline, Media, MediaType
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from loguru import logger


class BinaryGzipReaderFast(PipelineStep):
    name = "📒⚡ - Binary Gzip Media Reader (Fast/Threaded)"
    type = "Media Reader"

    def __init__(
        self,
        data_folder: DataFolderLike,
        workers: int = 1,  # Number of worker threads
        offset_byte_size: int = 4,  # Still needed to read length prefix
        block_size: int = 25*1024*1024,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.workers = workers
        self.offset_byte_size = offset_byte_size
        self.block_size = block_size
        # Initialize thread-local storage manager

    def _init_thread_local(self):
        """Initializes file pointer state for the current thread."""
        self.thread_local.current_filename = None
        self.thread_local.current_fp = None  # Raw file pointer
        self.thread_local.current_gzip_fp = None  # Gzip wrapper

    def _close_thread_local_fps(self):
        """Closes both file pointers stored in the current thread's local storage."""
        if hasattr(self.thread_local, "current_gzip_fp") and self.thread_local.current_gzip_fp:
            try:
                self.thread_local.current_gzip_fp.close()
            except Exception as e:
                logger.warning(f"Error closing thread-local gzip fp for {self.thread_local.current_filename}: {e}")

        if hasattr(self.thread_local, "current_fp") and self.thread_local.current_fp:
            try:
                self.thread_local.current_fp.close()
            except Exception as e:
                logger.warning(f"Error closing thread-local raw fp for {self.thread_local.current_filename}: {e}")

        self._init_thread_local()  # Reset state after closing

    def read_media_record(self, document: Document, media: Media) -> Document:
        """
        Reads a single media record using thread-local file pointers.
        Updates the media object in place and returns the document.
        """
        # Assuming the path to the compressed file is stored in media.path
        # If it's stored elsewhere (e.g., document.metadata), adjust accordingly.
        file_path = media.path
        offset = media.offset

        if file_path is None or offset is None:
            logger.warning(f"Media {media.id} in doc {document.id} missing path or offset, skipping.")
            media.metadata["read_error"] = "Missing path or offset"
            self.stat_update("media_error", value=1, unit="media")
            return document

        # Initialize thread-local storage if this is the first time the thread uses it
        if not hasattr(self.thread_local, "current_filename"):
            self._init_thread_local()

        gzip_fp = None
        with self.track_time():
            # Check if the current thread's fp matches the required file
            if (
                self.thread_local.current_filename == file_path
                and self.thread_local.current_fp
                and self.thread_local.current_gzip_fp
            ):
                # Reuse existing file pointers
                gzip_fp = self.thread_local.current_gzip_fp
            else:
                # Close the old file pointers if they exist for this thread
                self._close_thread_local_fps()
                # Open the new raw file (no compression handled by DataFolder)
                fp = self.data_folder.open(file_path, "rb", compression=None, block_size=self.block_size)
                # Open the gzip stream
                gzip.READ_BUFFER_SIZE = self.block_size
                gzip_fp = gzip.open(fp, "rb")
                # Store both new fps and filename in thread-local storage
                self.thread_local.current_fp = fp
                self.thread_local.current_gzip_fp = gzip_fp
                self.thread_local.current_filename = file_path

            # --- Core Reading Logic ---
            # Seek on the gzip stream (assuming offset is for the uncompressed stream)
            if gzip_fp.tell() != offset:
                print(f"seeking to {offset}")
                gzip_fp.seek(offset)
            # Get length of the record
            len_bytes = gzip_fp.read(self.offset_byte_size)
            if len(len_bytes) != self.offset_byte_size:
                    raise EOFError(f"Could not read length prefix ({len(len_bytes)}B/{self.offset_byte_size}B) at offset {offset} in {file_path}")
            length = int.from_bytes(len_bytes, "big")

            # Read the actual data
            content_bytes = gzip_fp.read(length)
            if len(content_bytes) != length:
                    raise EOFError(f"Could not read full record ({len(content_bytes)}B/{length}B) at offset {offset} in {file_path}")
            # --- End Reading Logic ---

            # Update the media object directly
            media.media_bytes = content_bytes

            self.stat_update("media_fetched", value=1, unit="media")
            self.stat_update("media_fetched_bytes", value=len(content_bytes), unit="bytes")
            # self.update_media_stats(media) # If you have this helper

        return document

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if data is None:
            return
        self.thread_local = threading.local()

        # Initialize thread-local storage for the main thread (used in finally)
        self._init_thread_local()

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = set()
            for document in data:
                # Submit one task per media item in the document
                for media_item in document.media:
                    # Basic check before submitting
                    if media_item.path and media_item.offset is not None:
                            # Keep the futures queue size manageable
                        while len(futures) >= 2*self.workers:
                            done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=None)
                            for future in done:
                                processed_doc = future.result()
                                yield processed_doc

                        new_future = executor.submit(self.read_media_record, document, media_item)
                        futures.add(new_future)
                    else:
                        logger.warning(f"Skipping media {media_item.id} in doc {document.id}: missing path or offset.")
                        media_item.metadata["read_error"] = "Missing path or offset before submission"
                        self.stat_update("media_error", value=1, unit="media")
                        yield document


            # Process remaining futures after input data is exhausted
            logger.info(f"Input data exhausted. Waiting for {len(futures)} remaining tasks.")
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=None)
                for future in done:
                    processed_doc = future.result()
                    yield processed_doc

            logger.info("Processing complete.")

            # Attempt cleanup for the main thread's FPs (workers cleanup on error/switch)
            self._close_thread_local_fps()
            logger.info("Attempted cleanup of main thread file pointers.")
