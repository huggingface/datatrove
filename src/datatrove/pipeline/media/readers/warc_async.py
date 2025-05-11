from datatrove.data import Document, DocumentsPipeline, Media, MediaType
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from warcio.archiveiterator import ArchiveIterator
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from loguru import logger

class WarcReaderAsync(PipelineStep):
    type = "Media Reader"
    name = "🌐 - Warc Reader Async"
    def __init__(self, data_folder: DataFolderLike):
        self.data_folder = get_datafolder(data_folder)
        # Initialize thread-local storage
        super().__init__()

    def _init_thread_local(self):
        """Initializes file pointer state for the current thread."""
        self.thread_local.current_fp = None
        self.thread_local.current_filename = None

    def _close_thread_local_fp(self):
        """Closes the file pointer stored in the current thread's local storage, if any."""
        if hasattr(self.thread_local, "current_fp") and self.thread_local.current_fp:
            try:
                self.thread_local.current_fp.close()
            except Exception as e:
                logger.warning(f"Error closing thread-local file pointer for {self.thread_local.current_filename}: {e}")
        self._init_thread_local() # Reset state after closing

    def update_record(self, record: Document, warc_record):
        id_ = warc_record.rec_headers["WARC-Record-ID"]
        url = warc_record.rec_headers.get("WARC-Target-URI", None)
        date = warc_record.rec_headers.get("WARC-Date", None)
        # handle older formats
        if url is None:
            url = dict(warc_record.rec_headers.headers)["uri"]
        if date is None:
            date = dict(warc_record.rec_headers.headers)["archive-date"]

        
        content_bytes = warc_record.content_stream().read()
        record.media.append(Media(
            id=id_,
            type=MediaType.DOCUMENT,
            media_bytes=content_bytes,
            url=url,
            metadata=dict(warc_record.rec_headers.headers) | {"date": date},
        ))

        self.stat_update("media_fetched", value=1, unit="documents")
        self.stat_update("media_fetched_bytes", value=len(content_bytes), unit="bytes")
        self.update_media_stats(record.media[-1])
        return record
        
    def read_warc_record(self, record: Document):
        # Ensures each thread handles its own file operations independently, reusing fp if possible.
        warc_file = record.metadata["warc_filename"]
        warc_record_offset = record.metadata["warc_record_offset"]

        # Initialize thread-local storage if this is the first time the thread uses it
        if not hasattr(self.thread_local, "current_filename"):
            self._init_thread_local()

        fp = None
        try:
            # track_time moved here to measure the actual threaded work
            with self.track_time():
                # Check if the current thread's fp matches the required file
                if self.thread_local.current_filename == warc_file and self.thread_local.current_fp:
                    # Reuse existing file pointer
                    fp = self.thread_local.current_fp
                else:
                    # Close the old file pointer if it exists for this thread
                    self._close_thread_local_fp()
                    # Open the new file
                    fp = self.data_folder.open(warc_file, "rb")
                    # Store the new fp and filename in thread-local storage
                    self.thread_local.current_fp = fp
                    self.thread_local.current_filename = warc_file

                # Seek to the correct offset using the (potentially reused) file pointer
                fp.seek(warc_record_offset)
                # Create a NEW ArchiveIterator instance - reusing iterators after seeking is unsafe
                ait = ArchiveIterator(fp)
                warc_record = next(ait)
                record = self.update_record(record, warc_record)
                # DO NOT close fp here if we want to reuse it
                # ait does not need explicit closing when created this way
            return record
        except Exception as e:
            logger.warning(f"Error reading WARC record for {record.id} from {warc_file} at offset {warc_record_offset}: {e}")
            # Mark the record with an error
            record.metadata["warc_read_error"] = str(e)
            # If an error occurred, close the potentially problematic file pointer for this thread
            self._close_thread_local_fp()
            return record # Return the record even if reading failed

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if data is None:
            return
        self.thread_local = threading.local()

        # Use a try/finally block to ensure thread-local resources are cleaned up,
        # although perfect cleanup across threads in executor shutdown isn't guaranteed.
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = set()
                for next_record in data:
                    # Keep the futures queue size manageable
                    while len(futures) >= self.workers * 2: # Keep queue size reasonable
                        done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=5) # Short timeout
                        for future in done:
                            try:
                                processed_record = future.result()
                                if "warc_read_error" not in processed_record.metadata:
                                    yield processed_record # Only yield successfully processed records
                            except Exception as e:
                                # Log unexpected errors during future processing
                                logger.error(f"Error processing future result: {e}")

                    # Submit the next record to be processed
                    if next_record.metadata.get("warc_filename") and next_record.metadata.get("warc_record_offset") is not None:
                        new_future = executor.submit(self.read_warc_record, next_record)
                        futures.add(new_future)
                    else:
                        logger.warning(f"Skipping record {next_record.id}: missing WARC metadata.")

                # Process remaining futures after input data is exhausted
                logger.info(f"Input data exhausted. Waiting for {len(futures)} remaining tasks.")
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0) # Longer timeout when waiting
                    for future in done:
                        try:
                            processed_record = future.result()
                            if "warc_read_error" not in processed_record.metadata:
                                 yield processed_record
                        except Exception as e:
                            logger.error(f"Error processing future result during final wait: {e}")

                logger.info("Processing complete.")
        finally:
            # Attempt to close any remaining file pointers in the main thread's local storage
            # Note: This doesn't explicitly close FPs held by worker threads if they haven't terminated cleanly,
            # but the executor shutdown should handle thread termination.
            self._close_thread_local_fp()
            logger.info("Attempted cleanup of main thread file pointer.")
