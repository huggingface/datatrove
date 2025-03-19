from collections import Counter, defaultdict
import gzip
from typing import IO, Any, Callable, Literal

from datatrove.data import Media
from datatrove.io import DataFolderLike
from datatrove.pipeline.media_writers.base_media_writer import BaseMediaWriter
import struct


class BinaryGzipWriter(BaseMediaWriter):
    default_output_filename: str = "${rank}.bin.gz"
    name = "📒 Binary Gzip"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        max_file_size: int = 5 * 2**30,  # 5GB
        offset_byte_size: int = 4,
    ):
        super().__init__(
            output_folder,
            output_filename,
            mode="wb",
            max_file_size=max_file_size,
        )
        self._writers = {}
        self.offset_byte_size = offset_byte_size

    def _on_file_switch(self, original_name, old_filename, new_filename):
        """
            Called when we are switching file from "old_filename" to "new_filename" (original_name is the filename
            without 000_, 001_, etc)
        Args:
            original_name: name without file counter
            old_filename: old full filename
            new_filename: new full filename
        """
        self._writers.pop(original_name).close()
        super()._on_file_switch(original_name, old_filename, new_filename)

    def _write(self, media: Media, file_handler: IO, filename: str):
        if filename not in self._writers:
            self._writers[filename] = gzip.open(file_handler, "wb")

        writer = self._writers[filename]
        offset = writer.tell()
        # write length of media bytes
        writer.write(len(media.media_bytes).to_bytes(self.offset_byte_size, "big"))
        # write media bytes
        writer.write(media.media_bytes)
        print(f"Wrote media {media.id} to {filename} at offset {offset} and length {len(media.media_bytes)}, writer type: {type(writer)}")
        return filename, offset

    def close(self):
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        super().close()
