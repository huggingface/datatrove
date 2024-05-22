# This file includes code from edugp/kenlm by Eduardo Gonzalez Ponferrada,
# licensed under the MIT License. The original code can be found at https://huggingface.co/edugp/kenlm.

import re
import unicodedata
from pathlib import Path
from typing import Dict

from huggingface_hub import hf_hub_url

from datatrove.io import cached_asset_path_or_download


MODEL_REPO = "edugp/kenlm"


class SentencePiece:
    def __init__(
        self,
        model_dataset: str,
        model_name: str,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dataset = model_dataset
        self._model = None

    @property
    def model(self):
        import sentencepiece

        if self._model is None:
            path = cached_asset_path_or_download(
                hf_hub_url(MODEL_REPO, str(Path(self.model_dataset, f"{self.model_name}.sp.model")))
            )
            self._model = sentencepiece.SentencePieceProcessor()
            self._model.load(path)
        return self._model

    def tokenize(self, text: dict) -> dict:
        tokenized = self.model.encode_as_pieces(text)
        return " ".join(tokenized)


class KenlmModel:
    digit_re: re.Pattern = re.compile(r"\d")
    unicode_punct: Dict[str, str] = {
        "，": ",",
        "。": ".",
        "、": ",",
        "„": '"',
        "”": '"',
        "“": '"',
        "«": '"',
        "»": '"',
        "１": '"',
        "」": '"',
        "「": '"',
        "《": '"',
        "》": '"',
        "´": "'",
        "∶": ":",
        "：": ":",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "；": ";",
        "–": "-",
        "—": " - ",
        "．": ". ",
        "～": "~",
        "’": "'",
        "…": "...",
        "━": "-",
        "〈": "<",
        "〉": ">",
        "【": "[",
        "】": "]",
        "％": "%",
        "►": "-",
    }
    unicode_punct_re = re.compile(f"[{''.join(unicode_punct.keys())}]")
    non_printing_chars_re = re.compile(f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]")

    def __init__(
        self,
        model_dataset: str,
        language: str,
        lower_case: bool = False,
        remove_accents: bool = False,
        normalize_numbers: bool = True,
        punctuation: int = 1,
    ):
        self.model_dataset = model_dataset
        self.language = language
        self.accent = remove_accents
        self.case = lower_case
        self.numbers = normalize_numbers
        self.punct = punctuation
        self._tokenizer = None
        self._model = None

    @property
    def model(self):
        import kenlm

        if self._model is None:
            model_path = Path(self.model_dataset, f"{self.language}.arpa.bin")
            path = cached_asset_path_or_download(hf_hub_url(MODEL_REPO, str(model_path)))
            self._model = kenlm.Model(path)
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = SentencePiece(self.model_dataset, self.language)
        return self._tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_dataset: str,
        language: str,
    ):
        return cls(
            model_dataset,
            language,
            False,
            False,
            True,
            1,
        )

    def pp(self, log_score, length):
        return 10.0 ** (-log_score / length)

    def get_perplexity(self, doc: str, normalize_cc_net: bool = True):
        if normalize_cc_net:
            doc = self.normalize(
                doc,
                accent=self.accent,
                case=self.case,
                numbers=self.numbers,
                punct=self.punct,
            )
        # Tokenize (after normalizing): See https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/mine.py#L352 for full pipeline
        doc = self.tokenizer.tokenize(doc)
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return round(self.pp(doc_log_score, doc_length), 1)

    def normalize(
        self,
        line: str,
        accent: bool = True,
        case: bool = True,
        numbers: bool = True,
        punct: int = 1,
    ) -> str:
        line = line.strip()
        if not line:
            return line
        if case:
            line = line.lower()
        if accent:
            line = self.strip_accents(line)
        if numbers:
            line = self.digit_re.sub("0", line)
        if punct == 1:
            line = self.replace_unicode_punct(line)
        elif punct == 2:
            line = self.remove_unicode_punct(line)
        line = self.remove_non_printing_char(line)
        return line

    def strip_accents(self, line: str) -> str:
        """Strips accents from a piece of text."""
        nfd = unicodedata.normalize("NFD", line)
        output = [c for c in nfd if unicodedata.category(c) != "Mn"]
        if len(output) == line:
            return line
        return "".join(output)

    def replace_unicode_punct(self, text: str) -> str:
        return "".join(self.unicode_punct.get(c, c) for c in text)

    def remove_unicode_punct(self, text: str) -> str:
        """More aggressive version of replace_unicode_punct but also faster."""
        return self.unicode_punct_re.sub("", text)

    def remove_non_printing_char(self, text: str) -> str:
        return self.non_printing_chars_re.sub("", text)
