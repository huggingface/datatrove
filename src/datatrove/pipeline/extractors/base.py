from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import StatHints


class BaseExtractor(PipelineStep):
    """Base Extractor module. Extractors extract text from html or other non-plain text formats"""

    type = "🛢 - EXTRAC"

    @abstractmethod
    def __init__(self, timeout: float = 0.1):
        """

        Args:
            timeout: the timeout for extraction, per document, in seconds
        """
        super().__init__()
        self.timeout = timeout
        self._warned_error = False

    @abstractmethod
    def extract(self, text: str) -> str:
        """abstract method that actually implements the extraction, e.g. trafilatura.

        Args:
          text: str: non-plain text

        Returns: extracted plain text

        """
        pass

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Iterates through each document in data and calls `timeout_extract` on it.

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:

        """
        executor = ProcessPoolExecutor(max_workers=1)
        try:
            for doc in data:
                self.stat_update(StatHints.total)
                with self.track_time():
                    # If submit fails, the pool was already broken from previous task
                    try:
                        future = executor.submit(self.extract, doc.text)
                    except BrokenProcessPool:
                        logger.warning(
                            "Found broken process pool, creating new executor and retrying current document"
                        )
                        executor.shutdown(wait=False)
                        executor = ProcessPoolExecutor(max_workers=1)
                        self.stat_update("broken_pool")
                        try:
                            future = executor.submit(self.extract, doc.text)
                        except BrokenProcessPool:
                            logger.error("New pool also broke, skipping document")
                            continue

                    try:
                        doc.text = future.result(timeout=self.timeout)
                        self.stat_update("extracted")
                    except TimeoutError:
                        future.cancel()
                        logger.warning("⏰ Timeout while cleaning record text. Skipping record.")
                        self.stat_update("timeout")
                        continue
                    except Exception as e:
                        future.cancel()
                        self.stat_update("clean_error")
                        if not self._warned_error:
                            logger.warning(
                                f'❌ Error "{e}" while cleaning record text. Skipping record. This message will only '
                                f"appear once."
                            )
                            self._warned_error = True
                        continue

                if doc.text:
                    self.stat_update(StatHints.forwarded)
                    self.update_doc_stats(doc)
                    yield doc
                else:
                    self.stat_update(StatHints.dropped)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
