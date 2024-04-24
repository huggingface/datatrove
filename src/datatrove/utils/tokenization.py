import itertools
import os.path
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
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
        self.tokenizer_name_or_path = None
        self.eos_token = None
        self._tokenizer: "Tokenizer" | None = None
        self._post_processor = None
        self._token_size = None
        self._token_format = None

    @property
    def token_size(self) -> int:
        if not self._token_size:
            self._token_size = 4 if self.tokenizer.get_vocab_size() > np.iinfo(np.uint16).max + 1 else 2
        return self._token_size

    @property
    def token_format(self) -> str:
        return "I" if self.token_size == 4 else "H"

    @property
    def tokenizer(self) -> "Tokenizer":
        if not self._tokenizer:
            if not self.tokenizer_name_or_path:
                raise ValueError("self.tokenizer_name_or_path needs to be set!")
            self._tokenizer = load_tokenizer(self.tokenizer_name_or_path)
            if self._post_processor:
                self._tokenizer.post_processor = self._post_processor
            elif self.eos_token:
                self._tokenizer.post_processor = TemplateProcessing(
                    single="$A <EOS>",
                    special_tokens=[("<EOS>", self.tokenizer.token_to_id(self.eos_token))],
                    pair=None,
                )
        return self._tokenizer


def batched(iterable, n):
    """In python 3.12+ we could use itertools.batched instead

    One difference with itertools.batched: we return a list instead of a tuple

    Args:
      iterable:
      n:

    Returns:

    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch
