from typing import Tuple

from datatrove.pipeline.enrishers.base_enrisher import BaseEnrisher
from datatrove.pipeline.filters.gopher_repetition_filter import find_duplicates
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


STOP_CHARS = (".", "'", '"', "!", "?")


class FineWebQualityEnrisher(BaseEnrisher):
    name = "🍷 FineWeb Quality Enrisher"

    def __init__(
        self,
        line_punct_ratio: bool = True,
        line_length: bool = True,
        char_duplicates_ratio: bool = True,
        new_line_ratio: bool = True,
        language: str = Languages.english,
        stop_chars: Tuple[str] = STOP_CHARS,
    ):
        super().__init__()
        self.line_punct_ratio = line_punct_ratio
        self.line_length = line_length
        self.char_duplicates_ratio = char_duplicates_ratio
        self.new_line_ratio = new_line_ratio
        self.tokenizer = load_word_tokenizer(language)
        self.stop_chars = stop_chars

    def enrish(self, doc) -> bool | tuple[bool, str]:
        lines = doc.text.split("\n")
        fineweb_metadata = {}
        self.stat_update("doc-total")
        if self.line_punct_ratio:
            line_punct_ratio = sum(1 for line in lines if line.endswith(self.stop_chars)) / len(lines)
            fineweb_metadata["line_punct_ratio"] = line_punct_ratio

        if self.line_length:
            line_length = [len(line) for line in lines]
            fineweb_metadata["line_length"] = line_length

        if self.char_duplicates_ratio:
            non_empty_lines = [line for line in lines if line.strip() != ""]
            fineweb_metadata["char_duplicates_ratio"] = find_duplicates(non_empty_lines)[1] / len(
                doc.text.replace("\n", "")
            )

        if self.new_line_ratio:
            words = self.tokenizer.word_tokenize(doc.text)
            new_line = doc.text.count("\n")
            fineweb_metadata["new_line_ratio"] = new_line / len(words)

        doc.metadata["fineweb"] = fineweb_metadata
        return doc
