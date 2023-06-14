import re

from datatrove import Document
from datatrove.pipeline.filters.base import BaseFilter, FilterResult


class RegexFilter(BaseFilter):

    def __init__(
            self,
            regex_exp: str,
            exclusion_reason: str | None = None,
            *args,
            **kwargs
    ):
        """
          filters if regex find at least one match

          @param regex_exp: regex expression
          """
        super(RegexFilter, self).__init__(*args, **kwargs)
        self.regex = re.compile(regex_exp)
        self.exclusion_reason = exclusion_reason

    def filter(self, doc: Document) -> FilterResult:
        """

        :param doc: document
        :return: is_filter
        """
        return FilterResult(not len(re.findall(self.regex, doc)) > 0, self.exclusion_reason)
