import gzip
from typing import IO

import zstandard as zstd

from datatrove.data import Media
from datatrove.io import DataFolderLike
from datatrove.pipeline.media.writers.base_media_writer import BaseMediaWriter


class BinaryZstdWriter(BaseMediaWriter):
    default_output_filename: str = "${rank}.bin.zst"
    name = "️️🗜️ - Binary Zstd"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        max_file_size: int = 5 * 2**30,  # 5GB
        length_byte_size: int = 4,
        compression_level: int = 3,
    ):
        super().__init__(
            output_folder,
            output_filename,
            mode="wb",
            max_file_size=max_file_size,
        )
        self.offset_byte_size = length_byte_size
        self.compressor = zstd.ZstdCompressor(level=compression_level)

    def _on_file_switch(self, original_name, old_filename, new_filename):
        """
            Called when we are switching file from "old_filename" to "new_filename" (original_name is the filename
            without 000_, 001_, etc)
        Args:
            original_name: name without file counter
            old_filename: old full filename
            new_filename: new full filename
        """
        super()._on_file_switch(original_name, old_filename, new_filename)

    def _write(self, media: Media, file_handler: IO, filename: str):
        # Store the starting offset of this record (length + data)
        record_start_offset = file_handler.tell()

        # Reserve space for the length of the compressed data
        file_handler.write(b'\0' * self.offset_byte_size)

        # Position after the reserved length space is where compressed data begins
        compressed_data_start_offset = file_handler.tell()

        # Stream compress the media_bytes directly to the file_handler
        # Each item will be its own zstd frame
        with self.compressor.stream_writer(file_handler, closefd=False) as zstd_writer:
            if media.media_bytes is not None: # Ensure media_bytes is not None
                zstd_writer.write(media.media_bytes)
            # The 'with' statement ensures zstd_writer is flushed and closed,
            # writing the zstd end-of-frame marker.

        # Get the position after the compressed data has been written
        compressed_data_end_offset = file_handler.tell()

        # Calculate the length of the compressed data
        compressed_length = compressed_data_end_offset - compressed_data_start_offset

        # Seek back to the beginning of the record to write the actual length
        file_handler.seek(record_start_offset)
        file_handler.write(compressed_length.to_bytes(self.offset_byte_size, "big"))

        # Seek forward to the end of the written compressed data, ready for the next record
        file_handler.seek(compressed_data_end_offset)

        return filename, record_start_offset # Return the offset to the start of the record

    def close(self):
        super().close()
