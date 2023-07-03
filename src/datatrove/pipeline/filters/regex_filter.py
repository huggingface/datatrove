import re

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter


class RegexFilter(BaseFilter):
    name = "🕵️ Regex"

    def __init__(self, regex_exp: str, **kwargs):
        """
        filters if regex find at least one match

        @param regex_exp: regex expression
        """
        super().__init__(**kwargs)
        self.regex = re.compile(regex_exp)

    def filter(self, doc: Document) -> bool:
        """

        :param doc: document
        :return: is_filter
        """
        return not len(self.regex.findall(doc.content)) > 0
