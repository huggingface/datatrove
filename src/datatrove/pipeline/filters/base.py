from abc import ABC, abstractmethod

from datatrove import DocumentsPipeline, Document
from datatrove.pipeline import PipelineStep

from dataclasses import dataclass


@dataclass
class FilterResult:
    is_kept: bool
    drop_reason: None | str


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
    def filter(self, doc: Document) -> FilterResult:
        """
        Filter modules' main method.
        Returns true if a sample should be filtered.

        @param doc: sample to (maybe) filter
        @return: bool - whether the doc should be filtered
        """
        return FilterResult(True, None)

    def step(self, docpipe: DocumentsPipeline) -> DocumentsPipeline:
        """
        step method for Filters.
        Drops documents that if .filter() is False

        @param datapipe: input DocumentsPipeline
        @return: DocumentsPipeline
        """

        for doc in docpipe:
            filter_result = self.filter(doc)
            if not filter_result.is_kept:
                continue
            yield doc
