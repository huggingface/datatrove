from abc import abstractmethod
import gzip
import heapq
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import IO

from datatrove.data import Document, DocumentsPipeline, Media, MediaType
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from loguru import logger

from datatrove.utils.typeshelper import StatHints


class BinaryReaderThreaded(PipelineStep):
    name = "📒⚡ - Binary Media Reader (Fast/Threaded)"
    type = "Media Reader"

    def __init__(
        self,
        data_folder: DataFolderLike,
        workers: int = 1,  # Number of worker threads
        preserve_order: bool = False,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.workers = workers
        self.preserve_order = preserve_order
        self.thread_local = None

    def _read_media_record_wrapper(self, document: Document, media: Media, task_index: int):
        media.media_bytes = self.read_media_record(media)
        self.update_media_stats(media)
        return document, task_index

    @abstractmethod
    def read_media_record(self, media: Media) -> bytes | None:
        """
        Reads a single media record using thread-local file pointers.
        Updates the media object in place and returns the document.
        """
        raise NotImplementedError

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        self.thread_local = threading.local()
        if data is None:
            return

        next_index = 0
        processed_task_heap = []
        task_index = 0

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = set()
            for document in data:
                # Submit one task per media item in the document
                for media_item in document.media:
                    # Basic check before submitting
                    if media_item.path and media_item.offset is not None:
                        # Keep the futures queue size manageable
                        while len(futures) >= 2*self.workers or (len(processed_task_heap) >= 2*self.workers):
                            done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=None)
                            for future in done:
                                processed_document, processed_task_index = future.result()
                                # push to heap
                                heapq.heappush(processed_task_heap, (processed_task_index, processed_document))
                            
                            while processed_task_heap and (not self.preserve_order or processed_task_heap[0][0] == next_index):
                                yield heapq.heappop(processed_task_heap)[1]
                                next_index += 1

                        new_future = executor.submit(self._read_media_record_wrapper, document, media_item, task_index)
                        futures.add(new_future)
                        task_index += 1
                    else:
                        logger.warning(f"Skipping media {media_item.id} in doc {document.id}: missing path or offset.")
                        media_item.metadata["read_error"] = "Missing path or offset before submission"
                        yield document

            # Process remaining futures after input data is exhausted
            logger.info(f"Input data exhausted. Waiting for {len(futures)} remaining tasks.")
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED, timeout=None)
                for future in done:
                    processed_document, processed_task_index = future.result()
                    heapq.heappush(processed_task_heap, (processed_task_index, processed_document))

                    # pop from heap while if index == next_index
                    while processed_task_heap and (not self.preserve_order or processed_task_heap[0][0] == next_index):
                        yield heapq.heappop(processed_task_heap)[1]
                        next_index += 1


