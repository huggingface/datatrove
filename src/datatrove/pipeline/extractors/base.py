import time
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
    def __init__(self, timeout, wamup_text):
        self.timeout = timeout
        self.process = None
        self.parent_conn = None
        self.wamup_text = wamup_text
        self.child_conn = None
        self.all_processes = []

    def set_oom_score_adj(self, score):
        if not -1000 <= score <= 1000:
            raise ValueError("Score must be between -1000 and +1000")
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(f"{score}\n")

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
        # Ensure this process is killed first
        self.set_oom_score_adj(1000)

        extract_fn(self.wamup_text)  # "warmup"
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

            deadline = time.monotonic() + self.timeout
            # loop with short sleeps instead of one big poll()
            while True:
                poll_timeout = max(0, min(5, deadline - time.monotonic() + 0.1))
                if self.parent_conn.poll(poll_timeout):
                    result = self.parent_conn.recv()
                    if isinstance(result, Exception):
                        raise result
                    return result

                # 2) Has the child died?
                if not self.process.is_alive():
                    raise EOFError("Child process died (probably OOM-killed)")

                # 3) Has our deadline passed?
                if time.monotonic() >= deadline:
                    raise TimeoutError("Document extraction timed out")
        except (TimeoutError, EOFError):
            self._cleanup_process()
            raise

    def _ensure_process(self, extract_fn):
        if self.process is None or not self.process.is_alive():
            if self.process is not None:
                self._cleanup_process()

            self.parent_conn, self.child_conn = Pipe()
            self.process = Process(target=self._worker, args=(self.child_conn, extract_fn), daemon=True)
            self.process.start()
            self.all_processes.append(self.process)
            # Wait for the "ready" signal from the worker process with a timeout
            if self.parent_conn.poll(self.timeout):
                self.parent_conn.recv()  # Receive the None signal
            else:
                # Timeout occurred before the worker process signaled it's ready
                self._cleanup_process()
                raise TimeoutError(
                    f"Worker process failed to initialize within {self.timeout} seconds (warmup timeout)."
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # First, clean up the current process
        self._cleanup_process()
        # Now clean up ALL processes
        alive_processes = [p for p in self.all_processes if p.is_alive()]
        logger.info(f"Found {len(alive_processes)} alive processes to terminate")

        if alive_processes:
            # Step 1: Terminate all processes
            for i, process in enumerate(alive_processes):
                try:
                    logger.debug(f"Terminating process {i + 1}/{len(alive_processes)}: {process.pid}")
                    process.terminate()
                except Exception as e:
                    logger.warning(f"Error terminating process {process.pid}: {e}")

            # Step 2: Give them time to terminate gracefully (longer timeout)
            logger.info("Waiting up to 5 seconds for processes to terminate gracefully...")
            for process in alive_processes:
                try:
                    process.join(timeout=5.0)  # Much longer timeout
                except Exception as e:
                    logger.warning(f"Error joining process {process.pid}: {e}")

            # Step 3: Kill any remaining processes
            still_alive_processes = [p for p in alive_processes if p.is_alive()]
            if still_alive_processes:
                logger.warning(f"Force killing {len(still_alive_processes)} stubborn processes...")
                for process in still_alive_processes:
                    try:
                        logger.debug(f"Killing process {process.pid}")
                        process.kill()
                        process.join(timeout=2.0)  # Final join with timeout
                    except Exception as e:
                        logger.error(f"Error killing process {process.pid}: {e}")

            # Step 4: Final check
            final_alive = [p for p in self.all_processes if p.is_alive()]
            if final_alive:
                logger.error(f"Failed to clean up {len(final_alive)} processes! PIDs: {[p.pid for p in final_alive]}")
            else:
                logger.info("Successfully cleaned up all processes")

        # Clear the list
        self.all_processes.clear()
