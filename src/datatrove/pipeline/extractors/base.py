import signal
from abc import abstractmethod

from loguru import logger

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


class BaseExtractor(PipelineStep):
    """Base Extractor module. Extractors extract text from html"""

    type = "🛢 - EXTRAC"

    @abstractmethod
    def __init__(self, timeout: float = 0.1):
        """
        :param timeout: the timeout for extraction, per document, in seconds
        """
        super().__init__()
        self.timeout = timeout

    @abstractmethod
    def extract(self, text: str) -> str:
        """abstract method that actually implements the extraction, e.g. trafilatura."""
        pass

    def timeout_extract(self, doc: Document):
        """Stops the extraction if it takes longer than timeout.
        This is the main entrypoint for this class.

        Args:
            doc: Document

        Returns:

        """

        def signal_handler(signum, frame):
            raise TimeoutError

        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            return self.extract(doc.text)

        except TimeoutError:
            logger.warning("⏰ Timeout while cleaning record tex.t Skipping record.")

        except Exception as e:
            logger.warning(f'❌ Error "{e}" while cleaning record text. Skipping record.')

        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            self.stat_update(StatHints.total)
            with self.track_time():
                doc.text = self.timeout_extract(doc)
            if doc.text:
                self.stat_update(StatHints.forwarded)
                self.update_doc_stats(doc)
                yield doc
            else:
                self.stat_update(StatHints.dropped)
