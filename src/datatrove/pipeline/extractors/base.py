from abc import abstractmethod
from multiprocessing import Pipe, Process

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import StatHints


class BaseExtractor(PipelineStep):
    """Base Extractor module. Extractors extract text from html or other non-plain text formats"""

    type = "🛢 - EXTRAC"

    @abstractmethod
    def __init__(self, timeout: float = 1):
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
        with ExtractorSandbox(timeout=self.timeout) as extractor:
            for doc in data:
                self.stat_update(StatHints.total)
                with self.track_time():
                    try:
                        doc.text = extractor.process_document(doc.text, self.extract)
                        self.stat_update("extracted")
                    except TimeoutError:
                        self.stat_update("timeout")
                        logger.warning("⏰ Timeout while cleaning record text. Skipping record.")
                        continue
                    except EOFError:
                        # Process died unexpectedly
                        self.stat_update("broken_process")
                        logger.warning("Process died unexpectedly, will create new process for next document")
                        continue
                    except Exception as e:
                        self.stat_update("clean_error")
                        if not self._warned_error:
                            logger.warning(
                                f'❌ Error "{e}" while cleaning record text. Skipping record. '
                                f"This message will only appear once."
                            )
                            self._warned_error = True
                        continue

                if doc.text:
                    self.stat_update(StatHints.forwarded)
                    self.update_doc_stats(doc)
                    yield doc
                else:
                    self.stat_update(StatHints.dropped)


class ExtractorSandbox:
    def __init__(self, timeout):
        self.timeout = timeout
        self.process = None
        self.parent_conn = None
        self.child_conn = None

    def _cleanup_process(self):
        if self.process is not None:
            self.parent_conn.close()
            self.child_conn.close()
            self.process.terminate()
            self.process.join(timeout=0.1)  # small clean up window
            if self.process.is_alive():
                self.process.kill()
            self.process = None
            self.parent_conn = None
            self.child_conn = None

    def _worker(self, conn, extract_fn):
        extract_fn("")  # "warmup"
        conn.send(None)  # ready
        while True:
            try:
                text = conn.recv()
                result = extract_fn(text)
                conn.send(result)
            except EOFError:
                break

    def process_document(self, text, extract_fn):
        self._ensure_process(extract_fn)
        try:
            self.parent_conn.send(text)
            if self.parent_conn.poll(timeout=self.timeout):
                result = self.parent_conn.recv()
                if isinstance(result, Exception):
                    raise result
                return result
            else:
                raise TimeoutError("Document extraction timed out")
        except (TimeoutError, EOFError):
            self._cleanup_process()
            raise

    def _ensure_process(self, extract_fn):
        if self.process is None or not self.process.is_alive():
            if self.process is not None:
                self._cleanup_process()

            self.parent_conn, self.child_conn = Pipe()
            self.process = Process(target=self._worker, args=(self.child_conn, extract_fn))
            self.process.start()
            self.parent_conn.recv()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_process()
        return False
