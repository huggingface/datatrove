from abc import ABC, abstractmethod
import signal
from loguru import logger

from datatrove.data import DocumentsPipeline, Document
from datatrove.pipeline.base import PipelineStep


class BaseExtractor(PipelineStep):
    type = "🛢️ - EXTRACTOR"

    @abstractmethod
    def __init__(self, timeout: float = 0.1, **kwargs):
        """
        Base Extractor module, it convert html to text.
        """
        super().__init__(**kwargs)
        self.timeout = timeout

    @abstractmethod
    def extract(self, doc: Document) -> bool:
        """
        abstract module that actually implements the extraction, e.g. trafilatura.
        """
        return True

    def timeout_extract(self, doc: Document):
        """
        stops the extraction if it takes longer than timeout
        :param doc: Documnet
        :return:
        """

        def signal_handler(signum, frame):
            raise TimeoutError

        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:

            return self.extract(doc)

        except TimeoutError:
            logger.warning(f"⏰ Timeout while cleaning record content. Skipping record.")
        except Exception as e:
            logger.warning(f"❌ Error \"{e}\" while cleaning record content. Skipping record.")

        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """

        """
        for doc in data:
            is_extracted = self.timeout_extract(doc)
            if is_extracted:
                yield doc
