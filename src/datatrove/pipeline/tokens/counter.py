from typing import TYPE_CHECKING

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


if TYPE_CHECKING:
    from tokenizers import Tokenizer


class TokensCounter(PipelineStep):
    """
    Counts how many tokens each document's text corresponds to
    """

    name = "📊 Counter"
    type = "🔢 - TOKENIZER"
    _requires_dependencies = ["tokenizers"]

    def __init__(
        self,
        tokenizer_name: str = "gpt2",  # tokenizer to use, from HF
        count_eos_token: bool = False,  # whether to count the EOS token on each document
    ):
        """

        Args:
            tokenizer_name: tokenizer to use (from HF)
            count_eos_token: whether to count EOS tokens as well (basically +1 per document)
        """
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.count_eos_token = count_eos_token
        self._tokenizer = None

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:

        """
        for document in data:
            count = len(self.tokenizer.encode(document.text).ids)
            self.stat_update("tokens", value=count)
            document.metadata["token_count"] = count
            yield document

    @property
    def tokenizer(self) -> "Tokenizer":
        if not self._tokenizer:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer


class LengthCounter(PipelineStep):
    """
    To be used after the TokensCounter block, counts how many samples have a given amount of tokens.
    Will absolutely spam the hell out of your stats.json
    """

    name = "📊 Document length counter"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
    ):
        super().__init__()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:

        """
        for document in data:
            count = document.metadata["token_count"]
            self.stats[count].update(1)
            yield document
