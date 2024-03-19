import os.path
from abc import ABC
from typing import TYPE_CHECKING

from tokenizers.processors import TemplateProcessing

from datatrove.pipeline.base import PipelineStep


if TYPE_CHECKING:
    from tokenizers import Tokenizer


def load_tokenizer(name_or_path: str) -> "Tokenizer":
    from tokenizers import Tokenizer

    if os.path.exists(name_or_path):
        return Tokenizer.from_file(name_or_path)
    return Tokenizer.from_pretrained(name_or_path)


class PipelineStepWithTokenizer(PipelineStep, ABC):
    _requires_dependencies = ["tokenizers"]

    def __init__(self):
        super().__init__()
        self._tokenizer = None
        self.tokenizer_name = None
        self._post_processor = None
        self.eos_token = None

    @property
    def tokenizer(self) -> "Tokenizer":
        if not self._tokenizer:
            if not self.tokenizer_name:
                raise ValueError("self.tokenizer_name needs to be set!")
            self._tokenizer: "Tokenizer" = load_tokenizer(self.tokenizer_name)
            if self._post_processor:
                self._tokenizer.post_processor = self._post_processor
            elif self.eos_token:
                self._tokenizer.post_processor = TemplateProcessing(
                    single="$A <EOS>",
                    special_tokens=[("<EOS>", self.tokenizer.token_to_id(self.eos_token))],
                    pair=None,
                )
        return self._tokenizer
