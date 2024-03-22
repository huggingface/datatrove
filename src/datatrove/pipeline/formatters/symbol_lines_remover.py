import re

from ...utils.text import PUNCTUATION
from .base import BaseFormatter


class SymbolLinesFormatter(BaseFormatter):
    """
    Removes lines that consist exclusively of symbols. Keeps lines that only have whitespace characters.
    """

    name = " ⚞ Symbol Lines Remover"

    def __init__(
        self,
        replace_char: str = "",  # change to \n to replace with a paragraph
    ):
        super().__init__()
        self.replace_char = replace_char
        puncts = "".join(map(re.escape, PUNCTUATION))
        self.symbol_regex = re.compile(rf"(^(([{puncts}]+[^\S\r\n]*)+\n?)+$((?<!\n)\n)?)", flags=re.MULTILINE)

    def format(self, text: str) -> str:
        return self.symbol_regex.sub(self.replace_char, text)
