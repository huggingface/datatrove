from datatrove.data import Document, DocumentsPipeline, Media, MediaType
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from warcio.archiveiterator import ArchiveIterator
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import heapq
from loguru import logger

class WarcReaderFast(PipelineStep):
    type = "Media Reader"
    name = "🌐 - Warc Reader Fast"
    def __init__(self, data_folder: DataFolderLike, workers: int = 5, block_size: int = 1024*128, preserve_order: bool = False):
        self.data_folder = get_datafolder(data_folder)
        self.workers = workers
        self.block_size = block_size
        self.preserve_order = preserve_order
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
        # If the media already exists update it
        if len(record.media) > 0:
            record.media[0].media_bytes = content_bytes
            record.media[0].metadata.update(dict(warc_record.rec_headers.headers) | {"date": date})
        else:
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
        
    def read_warc_record(self, record: Document, record_index: int):
        # Ensures each thread handles its own file operations independently, reusing fp if possible.
        warc_file = record.metadata["warc_filename"]
        warc_record_offset = record.metadata["warc_record_offset"]

        # Initialize thread-local storage if this is the first time the thread uses it
        if not hasattr(self.thread_local, "current_filename"):
            self._init_thread_local()

        fp = None
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
                # 64kb blocks
                fp = self.data_folder.open(warc_file, "rb", block_size=self.block_size)
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
        return record, record_index

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if data is None:
            return
        self.thread_local = threading.local()
        next_index = 0
        processed_record_heap = []

        try:
            with ThreadPoolExecutor(self.workers) as executor:
                futures = set()
                for record_index, record in enumerate(data):
                    # Keep the futures queue size manageable
                    while len(futures) >= self.workers * 2 or (len(processed_record_heap) >= self.workers * 2): # Keep queue size reasonable
                        done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=None) # Short timeout
                        for future in done:
                            processed_record, processed_record_index = future.result()
                            # push to heap
                            heapq.heappush(processed_record_heap, (processed_record_index, processed_record))
                        
                        while processed_record_heap and (not self.preserve_order or processed_record_heap[0][0] == next_index):
                            yield heapq.heappop(processed_record_heap)[1]
                            next_index += 1

                    # Submit the next record to be processed
                    if record.metadata.get("warc_filename") and record.metadata.get("warc_record_offset") is not None:
                        new_future = executor.submit(self.read_warc_record, record, record_index)
                        futures.add(new_future)
                    else:
                        logger.warning(f"Skipping record {record.id}: missing WARC metadata.")

                # Process remaining futures after input data is exhausted
                logger.info(f"Input data exhausted. Waiting for {len(futures)} remaining tasks.")
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=None) # Longer timeout when waiting
                    for future in done:
                        processed_record, processed_record_index = future.result()
                        heapq.heappush(processed_record_heap, (processed_record_index, processed_record))

                        # pop from heap while if index == next_index
                        while processed_record_heap and (not self.preserve_order or processed_record_heap[0][0] == next_index):
                            yield heapq.heappop(processed_record_heap)[1]
                            next_index += 1
        finally:
            # Attempt to close any remaining file pointers in the main thread's local storage
            # Note: This doesn't explicitly close FPs held by worker threads if they haven't terminated cleanly,
            # but the executor shutdown should handle thread termination.
            self._close_thread_local_fp()
            logger.info("Attempted cleanup of main thread file pointer.")
