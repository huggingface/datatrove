from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import gzip
from time import sleep
from typing import IO, Any, Callable, Literal

from datatrove.data import Document, DocumentsPipeline, Media, MediaType
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.media_readers.base_media_reader import BaseMediaReader
from loguru import logger


class SeekableMediaReader(BaseMediaReader, ABC):
    def __init__(self, data_folder: DataFolderLike):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.current_fp = None
        self.current_path = None

    @abstractmethod
    def open_file(self, path: str) -> IO:
        raise NotImplementedError

    @abstractmethod
    def close_file(self, fp: IO):
        raise NotImplementedError

    @abstractmethod
    def read_from_fp(self, fp: IO, media: Media) -> tuple[bytes | None, dict | None]:
        raise NotImplementedError

    def read_media(self, media: Media) -> tuple[bytes | None, dict | None]:
        if media.path is None:
            logger.warning(f"Media {media.id} has no path, skipping")
            return None, None

        if self.current_fp is None or self.current_path != media.path:
            if self.current_fp:
                self.close_file(self.current_fp)

            self.current_fp = self.open_file(media.path)
            self.current_path = media.path

        return self.read_from_fp(self.current_fp, media)

    def close(self):
        if self.current_fp:
            self.close_file(self.current_fp)



class BinaryGzipReader(SeekableMediaReader):
    default_output_filename: str = "${rank}.bin.gz"
    name = "📒 Binary Gzip"

    def __init__(
        self,
        data_folder: DataFolderLike,
        offset_byte_size: int = 4,
    ):
        super().__init__(data_folder)
        self.offset_byte_size = offset_byte_size

    def open_file(self, path: str) -> IO:
        self._fp = self.data_folder.open(path, compression=None)
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
        print(f"Reading media {media.id} from {media.offset} with length {length}, fp type: {type(fp)}")
        return fp.read(length), None

class WarcReader(PipelineStep):
    def __init__(self, data_folder: DataFolderLike = "s3://commoncrawl"):
        self.data_folder = get_datafolder(data_folder)
        self.current_fp = None
        self.current_file = None

    def update_record(self, record: Document, warc_record):
        content_bytes = warc_record.content_stream().read()
        record.media.append(Media(
            id=warc_record.rec_headers["WARC-Record-ID"],
            type=MediaType.DOCUMENT,
            media_bytes=content_bytes,
            url=warc_record.rec_headers["WARC-Target-URI"],
            metadata=dict(warc_record.rec_headers.headers),
        ))
        return record

        
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        from warcio.archiveiterator import ArchiveIterator
        if data is None:
            return

        for record in data:
            warc_file = record.metadata["warc_filename"]
            warc_record_offset = record.metadata["warc_record_offset"]

            if self.current_fp is None or self.current_file != warc_file:
                if self.current_fp:
                    self.current_fp.close()

                self.current_fp = self.data_folder.open(warc_file)
                self.current_file = warc_file

            # seek to the record offset
            self.current_fp.seek(warc_record_offset)

            # read the record
            ait = ArchiveIterator(self.current_fp)
            warc_record = next(ait)

            yield self.update_record(record, warc_record)

class HTTPFetchReader(PipelineStep):
    def __init__(self):
        self._scraper = None
        self._retry_delay = 2
        self._max_retries = 3

    @property
    def scraper(self):
        import cloudscraper
        if self._scraper is None:
            self._scraper = cloudscraper.create_scraper()
        return self._scraper

    def fetch_from_url(self, url: str) -> tuple[bytes | None, dict | None]:
        # Implement a retry mechanism with exponential backoff
        for attempt in range(self._max_retries):
            try:
                response = self.scraper.get(url)
                if response.status_code == 200:
                    return response.content, dict(response.headers)
                else:
                    logger.error(f"Failed to fetch media from {url}, status code: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching media from {url}: {e}")
            sleep(self._retry_delay ** attempt)
        return None, None
        

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if data is None:
            return

        for record in data:
            url = record.metadata["url"]
            print(f"Fetching media from {url}")

            # Fetch the media using cloudscraper
            response, headers = self.fetch_from_url(url)
            if response is not None:
                content_bytes = response
                record.media.append(Media(
                    id=url,
                    type=MediaType.DOCUMENT,
                    media_bytes=content_bytes,
                    url=url,
                    metadata=headers,
                ))
                yield record
            else:
                logger.error(f"Failed to fetch media from {url}")
