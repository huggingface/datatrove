import gzip
from typing import IO

from datatrove.data import Document, DocumentsPipeline, Media, MediaType
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.media.readers.base import BaseMediaReader
from loguru import logger


from datatrove.pipeline.media.readers.base import SeekableMediaReader


class BinaryGzipReader(SeekableMediaReader):
    name = "📒 - Binary Gzip Media Reader"
    type = "Media Reader"

    def __init__(
        self,
        data_folder: DataFolderLike,
        offset_byte_size: int = 4,
        # 1MB
        block_size: int = 20*1024*1024,
    ):
        super().__init__(data_folder)
        self.offset_byte_size = offset_byte_size
        self.block_size = block_size

    def open_file(self, path: str) -> IO:
        gzip.READ_BUFFER_SIZE = self.block_size
        self._fp = self.data_folder.open(path, compression=None, block_size=self.block_size)
        return gzip.open(self._fp, "rb")

    def close_file(self, fp: IO):
        if self._fp:
            self._fp.close()
        if fp:
            fp.close()
    
    def read_from_fp(self, fp: IO, media: Media) -> tuple[bytes | None, dict | None]:
        # seek to the record offset
        if media.offset is None:
            logger.warning(f"Media {media.id} has unknown offset, skipping")
            return None, None

        fp.seek(media.offset)
        # get length of the record
        bts = fp.read(self.offset_byte_size)
        length = int.from_bytes(bts, "big")
        return fp.read(length), None
