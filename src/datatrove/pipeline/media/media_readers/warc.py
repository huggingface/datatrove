import threading

from loguru import logger
from warcio.archiveiterator import ArchiveIterator

from datatrove.data import Media
from datatrove.pipeline.media.media_readers.base import BinaryReaderThreaded


class WarcReaderFast(BinaryReaderThreaded):
    type = "Media Reader"
    name = "🌐 - Warc Reader Fast"

    def read_media_record(self, media: Media):
        if media.offset is None or media.path is None:
            logger.warning(
                f"Thread {threading.current_thread().name}: Media {media.id} is missing offset or path, skipping."
            )
            return None

        # Ensures each thread handles its own file operations independently, reusing fp if possible.
        warc_file = media.path
        warc_record_offset = media.offset
        buff_size = media.length if media.length is not None else 1024 * 128

        # Initialize thread-local storage if this is the first time the thread uses it
        if self.thread_local is None:
            self.thread_local = threading.local()
        if not hasattr(self.thread_local, "current_filename"):
            self.thread_local.current_filename = None
            self.thread_local.current_fp = None

        fp = None
        # track_time moved here to measure the actual threaded work
        with self.track_time():
            # Check if the current thread's fp matches the required file
            if self.thread_local.current_filename == warc_file and self.thread_local.current_fp:
                # Reuse existing file pointer
                fp = self.thread_local.current_fp
            else:
                # Close the old file pointer if it exists for this thread
                if self.thread_local.current_fp:
                    try:
                        self.thread_local.current_fp.close()
                    except Exception as e:
                        logger.warning(
                            f"Error closing thread-local file pointer for {self.thread_local.current_filename}: {e}"
                        )
                # Open the new file
                fp = self.data_folder.open(warc_file, "rb", cache_type="none")
                # Store the new fp and filename in thread-local storage
                self.thread_local.current_fp = fp
                self.thread_local.current_filename = warc_file

            # Seek to the correct offset using the (potentially reused) file pointer
            fp.seek(warc_record_offset)
            # Create a NEW ArchiveIterator instance - reusing iterators after seeking is unsafe
            ait = ArchiveIterator(fp, block_size=buff_size)
            warc_record = next(ait)

        content = warc_record.content_stream().read()
        return content
