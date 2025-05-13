from abc import abstractmethod
from multiprocessing import Pipe, Process

from datatrove.data import DocumentsPipeline, Media
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from datatrove.data import MediaType
from datatrove.utils.typeshelper import StatHints
from datatrove.pipeline.extractors.base import ExtractorSandbox
from typing import Optional
from datatrove.pipeline.writers.disk_base import DiskWriter

class BaseMediaPredictor(PipelineStep):
    """Base Predictor module. Predictors predict metadata from media"""

    type = "🛢 - PREDICT"

    @abstractmethod
    def __init__(self, timeout: float = 300):
        """

        Args:
            timeout: the timeout for prediction, per document, in seconds
        """
        super().__init__()
        self.timeout = timeout

    @abstractmethod
    def predict(self, media_bytes: bytes | None) -> dict:
        """abstract method that actually implements the prediction, e.g. trafilatura.

        Args:
          media_bytes: bytes: media bytes

        Returns: predicted metadata

        """
        pass

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Iterates through each document in data and calls `predict` on it.

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)
        """
        with ExtractorSandbox(timeout=self.timeout, wamup_text=b"") as extractor:
            for doc in data:
                self.stat_update(StatHints.total)
                with self.track_time():
                    for media in doc.media:
                        try:
                            metadata = extractor.process_document(media.media_bytes, self.predict)
                            if metadata.get("prediction_error"):
                                media.metadata["prediction_failed"] = metadata["prediction_error"]
                                continue

                            media.metadata["pdf_metadata"] = metadata
                            self.stat_update("predicted")
                        except TimeoutError:
                            self.stat_update("timeout")
                            logger.warning("⏰ Timeout while predicting metadata. Skipping record.")
                            media.metadata["prediction_failed"] = "timeout"
                            continue
                        except EOFError:
                            # Process died unexpectedly
                            self.stat_update("broken_process")
                            logger.warning("Process died unexpectedly, will create new process for next document")
                            media.metadata["prediction_failed"] = "broken_process"
                            continue
                        except Exception as e:
                            self.stat_update("prediction_error")
                            if not self._warned_error:
                                logger.warning(
                                    f'❌ Error "{e}" while predicting metadata. Skipping record. '
                                    f"This message will only appear once."
                                )
                                self._warned_error = True
                            # Add this to metadata of the media
                            media.metadata["prediction_failed"] = str(e)
                            continue

                if all(media.metadata.get("prediction_failed", False) for media in doc.media):
                    for media in doc.media:
                        logger.warning(f"❌ Media {media.id} prediction failed with error {media.metadata['prediction_failed']}")
                    self.stat_update(StatHints.dropped)
                    yield doc
                else:
                    self.stat_update(StatHints.forwarded)
                    yield doc


