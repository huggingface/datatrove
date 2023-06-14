from abc import ABC, abstractmethod

from datatrove import DocumentsPipeline, Document
from datatrove.pipeline import PipelineStep


class BaseFilter(PipelineStep, ABC):
    @abstractmethod
    def __init__(
            self,
            *args,
            **kwargs
    ):
        """
        Abstract Filter pipeline step.
        A Filter drops samples

        @param args:
        @param kwargs:
        """
        super(BaseFilter, self).__init__(*args, **kwargs)

    @abstractmethod
    def filter(self, doc: Document) -> bool:
        """
        Filter modules' main method.
        Returns true if a sample should be filtered.

        @param doc: sample to (maybe) filter
        @return: bool - whether the doc should be filtered
        """
        return False

    def step(self, docpipe: DocumentsPipeline) -> DocumentsPipeline:
        """
        step method for Filters.
        Drops documents that trigger the filter method.

        @param datapipe: input DocumentsPipeline
        @return: DocumentsPipeline
        """

        for doc in docpipe:
            is_filter = self.filter(doc)
            if is_filter:
                continue
            yield doc
