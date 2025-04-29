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


def chunk_doc_ends(doc_ends, shuffle_chunk_size):
    all_chunks_doc_ends = []
    doc_end_i = 0
    for chunk_end in range(shuffle_chunk_size, doc_ends[-1] + 1, shuffle_chunk_size):
        chunk_doc_ends = []
        while doc_end_i < len(doc_ends) and doc_ends[doc_end_i] <= chunk_end:
            if doc_ends[doc_end_i] < chunk_end:  # avoid adding it twice
                chunk_doc_ends.append(doc_ends[doc_end_i])
            doc_end_i += 1
        chunk_doc_ends.append(chunk_end)
        all_chunks_doc_ends.append(chunk_doc_ends)
    return all_chunks_doc_ends


class PipelineStepWithTokenizer(PipelineStep, ABC):
    _requires_dependencies = ["tokenizers"]

    def __init__(
        self,
        tokenizer_name_or_path: str | None = None,
        eos_token: str | None = None,
        post_processor: Optional[TemplateProcessing] = None,
    ):
        """Initialize the PipelineStepWithTokenizer.

        Args:
            tokenizer_name_or_path: Path or name of the tokenizer to load.
            eos_token: Optional EOS token to add via post-processing.
            post_processor: Optional custom post-processor.
        """
        super().__init__()
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.eos_token = eos_token
        self.post_processor = post_processor

    @cached_property
    def token_size(self) -> int:
        """Determine the token size in bytes based on the tokenizer vocabulary size.

        Returns:
            int: 4 if vocab size exceeds uint16 max, otherwise 2.
        """
        return 4 if self.tokenizer.get_vocab_size() > np.iinfo(np.uint16).max + 1 else 2

    @cached_property
    def token_format(self) -> str:
        """Get the struct format string corresponding to the token size.

        Returns:
            str: "I" for 4 bytes (uint32), "H" for 2 bytes (uint16).
        """
        return "I" if self.token_size == 4 else "H"

    @cached_property
    def tokenizer(self) -> "Tokenizer":
        """Load and configure the tokenizer instance.

        Raises:
            ValueError: If `tokenizer_name_or_path` is not set.

        Returns:
            "Tokenizer": The loaded tokenizer instance with optional post-processing.
        """
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
