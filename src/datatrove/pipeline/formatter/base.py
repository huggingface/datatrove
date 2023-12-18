from abc import ABC, abstractmethod

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


class BaseFormatter(PipelineStep, ABC):
    """Base module for Formatters. Formatters modify/remove specific sections of document text contents."""

    type = "💇‍♀️ - FORMAT"

    @abstractmethod
    def format(self, doc: Document) -> str:
        """Formatter modules main method.
        Returns the new text to set as main content.

        Args:
            doc: sample to format

        Returns:
            str - the new text for this sample
        """
        raise NotImplementedError

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            self.stat_update(StatHints.total)
            with self.track_time():
                formatted = self.format(doc)
                if formatted != doc.content:
                    self.stat_update("formatted")
            self.update_doc_stats(doc)
            yield doc
