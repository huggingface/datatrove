import signal
from abc import abstractmethod

from loguru import logger

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


class BaseExtractor(PipelineStep):
    type = "🛢️- EXTRAC"

    @abstractmethod
    def __init__(self, timeout: float = 0.1, **kwargs):
        """
        Base Extractor module, converts html to text.
        """
        super().__init__(**kwargs)
        self.timeout = timeout

    @abstractmethod
    def extract(self, content: str) -> str:
        """
        abstract module that actually implements the extraction, e.g. trafilatura.
        """
        pass

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
            return self.extract(doc.content)

        except TimeoutError:
            logger.warning("⏰ Timeout while cleaning record content. Skipping record.")

        except Exception as e:
            logger.warning(f'❌ Error "{e}" while cleaning record content. Skipping record.')

        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """ """
        for doc in data:
            self.stat_update(StatHints.total)
            with self.stats.time_manager:
                doc.content = self.timeout_extract(doc)
            if doc.content:
                self.stat_update(StatHints.forwarded)
                self.stats.doc_len.update(len(doc.content))
                yield doc
            else:
                self.stat_update(StatHints.dropped)
