from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor

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

    @abstractmethod
    def extract(self, text: str) -> str:
        """abstract method that actually implements the extraction, e.g. trafilatura.

        Args:
          text: str: non-plain text

        Returns: extracted plain text

        """
        pass

    def clean_html(self, html: str) -> str:
        """Default implementation of `clean_html` for extractors that don't return a cleaned version of the HTML

        Since not all extractors produce a cleaned version of the HTML as a part of the extraction process,
        this default implementation throws a warning and simply returns the original HTML string.

        Args:
            html: str: the HTML content to clean

        Returns:
            str: the cleaned HTML
        """
        return html

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Iterates through each document in data and calls `timeout_extract` on it.

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:

        """
        with ThreadPoolExecutor() as executor:  # more reliable than using signal for timeouts
            for doc in data:
                self.stat_update(StatHints.total)
                with self.track_time():
                    future = executor.submit(self.extract, doc.text)
                    try:
                        doc.text = future.result(timeout=self.timeout)
                    except TimeoutError:
                        logger.warning(
                            "⏰ Timeout while cleaning record text. Skipping record.")
                        continue
                    except Exception as e:
                        logger.warning(
                            f'❌ Error "{e}" while cleaning record text. Skipping record.')
                        continue
                if doc.text:
                    self.stat_update(StatHints.forwarded)
                    self.update_doc_stats(doc)
                    yield doc
                else:
                    self.stat_update(StatHints.dropped)
