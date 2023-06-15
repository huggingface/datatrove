import re

from datatrove import Document
from datatrove.pipeline.filters.base import BaseFilter, FilterResult


class RegexFilter(BaseFilter):

    def __init__(
            self,
            regex_exp: str,
            exclusion_reason: str | None = None,
            **kwargs
    ):
        """
          filters if regex find at least one match

          @param regex_exp: regex expression
          """
        super().__init__(**kwargs)
        self.regex = re.compile(regex_exp)
        self.exclusion_reason = exclusion_reason

    def filter(self, doc: Document) -> bool:
        """

        :param doc: document
        :return: is_filter
        """
        return not len(self.regex.findall(doc)) > 0
