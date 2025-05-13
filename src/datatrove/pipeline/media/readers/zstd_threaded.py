import threading

import zstandard as zstd
from loguru import logger

from datatrove.data import Document, Media
from datatrove.io import DataFolderLike
from datatrove.pipeline.media.readers.base_threaded import BinaryReaderThreaded


class ZstdThreadedReader(BinaryReaderThreaded):
    name = "📒⚡ Zstd Media Reader (Streaming/Threaded)"
    type = "Media Reader"

    def __init__(
        self,
        data_folder: DataFolderLike,
        workers: int = 1,
        offset_byte_size: int = 4,
    ):
        super().__init__(data_folder, workers)
        self.offset_byte_size = offset_byte_size

    def _init_thread_local(self):
        """Initializes file pointer state for the current thread."""
        self.thread_local.current_filename = None
        self.thread_local.current_fp = None
        # We use just single decompressor for single thread as each file is ended with frame flush which results
        # in decompressor state reset.
        self.thread_local.current_zstd_decompressor = zstd.ZstdDecompressor(format=zstd.FORMAT_ZSTD1_MAGICLESS)

    def _close_thread_local_fp(self):
        """Closes the file pointer stored in the current thread's local storage."""
        if self.thread_local.current_fp:
            self.thread_local.current_fp.close()
            self.thread_local.current_fp = None
            self.thread_local.current_filename = None

    def read_media_record(self, media: Media) -> bytes | None:
        current_thread_name = threading.current_thread().name
        file_path = media.path
        offset = media.offset

        if file_path is None or offset is None:
            logger.warning(
                f"Thread {current_thread_name}: Media {media.id} is missing path or offset, skipping."
            )
            return None

        # Initialize thread-local storage if this is the first time the thread uses it
        if not hasattr(self.thread_local, "current_filename"):
            self._init_thread_local()

        if (
            self.thread_local.current_filename != file_path
            or not self.thread_local.current_fp
            or self.thread_local.current_fp.closed
        ):
            self._close_thread_local_fp()  # Close previous file if any
            self.thread_local.current_fp = self.data_folder.open(
                file_path, "rb"
            )
            self.thread_local.current_filename = file_path

        # --- Core Reading Logic ---
        fp = self.thread_local.current_fp
        fp.seek(offset) # Seek to the start of the length prefix
        length_bytes = fp.read(self.offset_byte_size)
        if length_bytes is None or len(length_bytes) != self.offset_byte_size:
            raise ValueError(f"Thread {current_thread_name}: Media {media.id} is missing length prefix, skipping.")
        length = int.from_bytes(length_bytes, "big")

        # Read the length prefix
        with self.thread_local.current_zstd_decompressor.stream_reader(fp, read_across_frames=False, closefd=False) as zstd_stream_reader:
            return zstd_stream_reader.read(length)
