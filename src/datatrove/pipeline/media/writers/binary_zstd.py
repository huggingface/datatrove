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
        compression_level: int = 3,
        offset_byte_size: int = 4,

    ):
        super().__init__(
            output_folder,
            output_filename,
            mode="wb",
            max_file_size=max_file_size,
        )
        self.offset_byte_size = offset_byte_size
        self.compression_level = compression_level
        self.zstd_compressor = zstd.ZstdCompressor(compression_params=zstd.ZstdCompressionParameters.from_level(self.compression_level, format=zstd.FORMAT_ZSTD1_MAGICLESS,
                    write_checksum=0, write_content_size=0))
    def _write(self, media: Media, file_handler: IO, filename: str):
        if media.media_bytes is None:
            raise ValueError(f"Media {media.id} has no media bytes")

        record_start_offset = file_handler.tell()

        # Write the uncompressed size
        file_handler.write(len(media.media_bytes).to_bytes(self.offset_byte_size, "big"))
        with self.zstd_compressor.stream_writer(file_handler, closefd=False, size=len(media.media_bytes)) as zstd_writer:
            zstd_writer.write(media.media_bytes)

        return filename, record_start_offset # Return the offset to the start of the record

    def close(self):
        super().close()
