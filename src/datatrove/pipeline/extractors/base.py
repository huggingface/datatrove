from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor

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
        with ProcessPoolExecutor(
            max_workers=1
        ) as executor:  # Single process. We use ProcessPool instead of ThreadPool
            # to be able to kill timed out tasks, lest them continue running for minutes and OOM
            for doc in data:
                self.stat_update(StatHints.total)
                with self.track_time():
                    future = executor.submit(self.extract, doc.text)
                    try:
                        doc.text = future.result(timeout=self.timeout)
                    except TimeoutError:
                        future.cancel()  # Kill the timed-out process
                        logger.warning("⏰ Timeout while cleaning record text. Skipping record.")
                        continue
                    except Exception as e:
                        future.cancel()  # Good practice to cancel on other errors too
                        if not self._warned_error:
                            logger.warning(
                                f'❌ Error "{e}" while cleaning record text. Skipping record. This message will only appear once.'
                            )
                            self._warned_error = True
                        continue

                if doc.text:
                    self.stat_update(StatHints.forwarded)
                    self.update_doc_stats(doc)
                    yield doc
                else:
                    self.stat_update(StatHints.dropped)
