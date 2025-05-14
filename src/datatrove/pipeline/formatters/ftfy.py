from .base import BaseFormatter


class FTFYFormatter(BaseFormatter):
    name = "😎 FTFY"
    _requires_dependencies = ["ftfy"]

    def format(self, text: str) -> str:
        import ftfy

        return ftfy.fix_text(text)
