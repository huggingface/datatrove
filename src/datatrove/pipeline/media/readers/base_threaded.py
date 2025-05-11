import gzip
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import IO

from datatrove.data import Document, DocumentsPipeline, Media, MediaType
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from loguru import logger

from libs.datatrove.src.datatrove.utils.typeshelper import StatHints


class BinaryGzipReaderFast(PipelineStep):
    name = "📒⚡ - Binary Gzip Media Reader (Fast/Threaded)"
    type = "Media Reader"

    def __init__(
        self,
        data_folder: DataFolderLike,
        workers: int = 1,  # Number of worker threads
        offset_byte_size: int = 4,  # Still needed to read length prefix
        block_size: int = 25*1024*1024,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.workers = workers
        self.offset_byte_size = offset_byte_size
        self.block_size = block_size

    def read_media_record(self, document: Document, media: Media) -> Document:
        """
        Reads a single media record using thread-local file pointers.
        Updates the media object in place and returns the document.
        """
        pass

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if data is None:
            return

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = set()
            for document in data:
                # Submit one task per media item in the document
                for media_item in document.media:
                    # Basic check before submitting
                    if media_item.path and media_item.offset is not None:
                            # Keep the futures queue size manageable
                        while len(futures) >= 2*self.workers:
                            done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=None)
                            for future in done:
                                processed_doc = future.result()
                                self.stat_update(StatHints.total)
                                self.update_media_stats(processed_doc.media)
                                yield processed_doc

                        new_future = executor.submit(self.read_media_record, document, media_item)
                        futures.add(new_future)
                    else:
                        logger.warning(f"Skipping media {media_item.id} in doc {document.id}: missing path or offset.")
                        media_item.metadata["read_error"] = "Missing path or offset before submission"
                        self.stat_update("media_error", value=1, unit="media")
                        yield document


            # Process remaining futures after input data is exhausted
            logger.info(f"Input data exhausted. Waiting for {len(futures)} remaining tasks.")
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=None)
                for future in done:
                    processed_doc = future.result()
                    yield processed_doc

