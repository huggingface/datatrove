import re

from datatrove import Document
from datatrove.pipeline.filters.base import BaseFilter


class RegexFilter(BaseFilter):

    def __init__(
            self,
            regex_exp: str,
            *args,
            **kwargs
    ):
        """
          filters if regex find at least one match

          @param regex_exp: regex expression
          """
        super(RegexFilter, self).__init__(*args, **kwargs)
        self.regex_exp = regex_exp

    def filter(self, doc: Document) -> bool:
        """

        :param doc: document
        :return: is_filter
        """
        return True if len(re.findall(self.regex_exp, doc)) > 0 else False
