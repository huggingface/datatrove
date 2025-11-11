from typing import IO

import zstandard as zstd

from datatrove.data import Media
from datatrove.io import DataFolderLike
from datatrove.pipeline.media.media_writers.base import BaseMediaWriter


class ZstdWriter(BaseMediaWriter):
    default_output_filename: str = "${rank}.bin.zst"
    name = "️️🗜️ - Binary Zstd"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        max_file_size: int = 5 * 2**30,  # 5GB
        compression_level: int = 3,
    ):
        super().__init__(
            output_folder,
            output_filename,
            mode="wb",
            max_file_size=max_file_size,
        )
        self.compression_level = compression_level

    @property
    def compressor(self):
        if not hasattr(self, "_compressor"):
            self._compressor = zstd.ZstdCompressor(
                compression_params=zstd.ZstdCompressionParameters.from_level(
                    self.compression_level, format=zstd.FORMAT_ZSTD1_MAGICLESS, write_checksum=0, write_content_size=0
                )
            )
        return self._compressor

    def _write(self, media: Media, file_handler: IO, filename: str):
        if media.media_bytes is None:
            raise ValueError(f"Media {media.id} has no media bytes")

        record_start_offset = file_handler.tell()

        # Write the uncompressed size
        with self.compressor.stream_writer(file_handler, closefd=False, size=len(media.media_bytes)) as zstd_writer:
            zstd_writer.write(media.media_bytes)

        compressed_size = file_handler.tell() - record_start_offset
        return filename, record_start_offset, compressed_size

    def close(self):
        super().close()
