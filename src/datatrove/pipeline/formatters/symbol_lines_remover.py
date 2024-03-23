from ...utils.text import PUNCTUATION_SET
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
        # loop actually seems faster
        # puncts = "".join(map(re.escape, PUNCTUATION))
        # self.symbol_regex = re.compile(rf"(^(([{puncts}]+[^\S\r\n]*)+\n?)+$((?<!\n)\n)?)", flags=re.MULTILINE)

    def format(self, text: str) -> str:
        formatted = []
        in_removed_span = False
        for line in text.splitlines():
            chars_line = line.strip() != "" and all(c in PUNCTUATION_SET or c == " " for c in line)
            if chars_line and not in_removed_span:
                if self.replace_char:
                    formatted.append(self.replace_char)
                in_removed_span = True
            elif not chars_line:
                formatted.append(line)
                in_removed_span = False
        return "\n".join(formatted)
