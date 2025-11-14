import threading
from typing import IO

import zstandard as zstd
from loguru import logger

from datatrove.data import Media
from datatrove.io import DataFolderLike
from datatrove.pipeline.media.media_readers.base import BinaryReaderThreaded


class LimitedReader(IO):
    """A wrapper around a file-like object that limits the number of bytes that can be read."""

    def __init__(self, fp: IO, max_bytes: int):
        self.fp = fp
        self.max_bytes = max_bytes
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        """Read at most size bytes, but never exceed the max_bytes limit."""
        if self.bytes_read >= self.max_bytes:
            return b""

        # Calculate how many bytes we can actually read
        if size == -1:
            # Read all remaining bytes up to the limit
            remaining = self.max_bytes - self.bytes_read
        else:
            # Read the minimum of requested size and remaining bytes
            remaining = min(size, self.max_bytes - self.bytes_read)

        if remaining <= 0:
            return b""

        data = self.fp.read(remaining)
        self.bytes_read += len(data)
        return data

    def close(self):
        """Close the underlying file pointer."""
        if hasattr(self.fp, "close"):
            self.fp.close()

    def seek(self, offset: int, whence: int = 0):
        """Seek in the underlying file pointer."""
        return self.fp.seek(offset, whence)

    def tell(self):
        """Get current position in the underlying file pointer."""
        return self.fp.tell()


class ZstdReader(BinaryReaderThreaded):
    name = "📒⚡ Zstd Media Reader (Streaming/Threaded)"
    type = "Media Reader"

    def __init__(
        self,
        data_folder: DataFolderLike,
        block_size: int = 20 * 1024 * 1024,
        workers: int = 1,
        offset_byte_size: int = 4,
        preserve_order: bool = False,
    ):
        super().__init__(data_folder, workers, preserve_order)
        self.offset_byte_size = offset_byte_size
        self.block_size = block_size

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

    def _read_without_length(
        self,
        zstd_decompressor_reader: zstd.ZstdDecompressor,
        media_id: str,
        offset: int,
        current_thread_name: str,
        fp: IO,
    ) -> bytes | None:
        """
        Read the media without the length prefix. Used in v1 of the zstd writer, however it's slower as it requires an extra read and we don't know the length of the media.
        Args:
            zstd_decompressor_reader: The zstd decompressor reader.
            media_id: The media id.
            offset: The offset to the start of the media.
            current_thread_name: The current thread name.
            fp: The file pointer.
        """
        fp.seek(offset)  # Seek to the start of the length prefix
        length_bytes = fp.read(self.offset_byte_size)
        if length_bytes is None or len(length_bytes) != self.offset_byte_size:
            raise ValueError(f"Thread {current_thread_name}: Media {media_id} is missing length prefix, skipping.")
        content_length = int.from_bytes(length_bytes, "big")

        # Read the length prefix
        # Has read_size parameter (== block size)
        with zstd_decompressor_reader.stream_reader(fp, read_across_frames=False, closefd=False) as zstd_stream_reader:
            return zstd_stream_reader.read(content_length)

    def _read_with_length(
        self, zstd_decompressor_reader: zstd.ZstdDecompressor, offset: int, length: int, fp: IO
    ) -> bytes | None:
        """
        Read the media with the length prefix. Used in v2 of the zstd writer.
        Args:
            zstd_decompressor_reader: The zstd decompressor reader.
            media_id: The media id.
            offset: The offset to the start of the media.
            length: The length of the media.
            fp: The file pointer.
        """
        fp.seek(offset)
        with zstd_decompressor_reader.stream_reader(
            LimitedReader(fp, length), read_across_frames=False, closefd=False
        ) as zstd_stream_reader:
            return zstd_stream_reader.read()

    def read_media_record(self, media: Media) -> bytes | None:
        if media.offset is None or media.path is None:
            logger.warning(
                f"Thread {threading.current_thread().name}: Media {media.id} is missing offset or path, skipping."
            )
            return None

        current_thread_name = threading.current_thread().name
        file_path = media.path
        offset = media.offset
        length = media.length

        if file_path is None or offset is None:
            logger.warning(f"Thread {current_thread_name}: Media {media.id} is missing path or offset, skipping.")
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
            # ADD block size here
            self.thread_local.current_fp = self.data_folder.open(file_path, "rb", block_size=self.block_size)
            self.thread_local.current_filename = file_path

        # --- Core Reading Logic ---
        fp = self.thread_local.current_fp
        if length is None:
            return self._read_without_length(
                self.thread_local.current_zstd_decompressor, media.id, offset, current_thread_name, fp
            )
        else:
            return self._read_with_length(self.thread_local.current_zstd_decompressor, offset, length, fp)
