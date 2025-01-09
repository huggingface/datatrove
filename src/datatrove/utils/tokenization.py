import os.path
from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING, Optional

import numpy as np
from tokenizers.processors import TemplateProcessing

from datatrove.pipeline.base import PipelineStep


if TYPE_CHECKING:
    from tokenizers import Tokenizer


def load_tokenizer(name_or_path: str) -> "Tokenizer":
    from tokenizers import Tokenizer

    if os.path.isfile(name_or_path):
        return Tokenizer.from_file(name_or_path)

    return Tokenizer.from_pretrained(name_or_path)


class PipelineStepWithTokenizer(PipelineStep, ABC):
    _requires_dependencies = ["tokenizers"]

    def __init__(
        self,
        tokenizer_name_or_path: str | None = None,
        eos_token: str | None = None,
        post_processor: Optional[TemplateProcessing] = None,
    ):
        super().__init__()
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.eos_token = eos_token
        self.post_processor = post_processor

    @cached_property
    def token_size(self) -> int:
        return 4 if self.tokenizer.get_vocab_size() > np.iinfo(np.uint16).max + 1 else 2

    @cached_property
    def token_format(self) -> str:
        return "I" if self.token_size == 4 else "H"

    @cached_property
    def tokenizer(self) -> "Tokenizer":
        if not self.tokenizer_name_or_path:
            raise ValueError("self.tokenizer_name_or_path needs to be set!")
        tokenizer = load_tokenizer(self.tokenizer_name_or_path)
        if self.post_processor:
            tokenizer.post_processor = self.post_processor
        elif self.eos_token:
            tokenizer.post_processor = TemplateProcessing(
                single="$A <EOS>",
                special_tokens=[("<EOS>", tokenizer.token_to_id(self.eos_token))],
                pair=None,
            )
        return tokenizer
