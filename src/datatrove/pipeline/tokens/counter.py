from loguru import logger

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


TOKENIZERS_INSTALLED = True
try:
    from tokenizers import Tokenizer
except ImportError:
    TOKENIZERS_INSTALLED = False


class TokensCounter(PipelineStep):
    name = "📊 Counter"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        tokenizer_name: str = "gpt2",  # tokenizer to use, from HF
        count_eos_token: bool = False,  # whether to count the EOS token on each document
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.count_eos_token = count_eos_token
        self._tokenizer = None

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for document in data:
            count = len(self.tokenizer.encode(document.text).ids)
            self.stat_update("tokens", value=count)
            document.metadata["token_count"] = count
            yield document

    @property
    def tokenizer(self) -> Tokenizer:
        if not self._tokenizer:
            if not TOKENIZERS_INSTALLED:
                logger.error("`tokenizers` is required to run DocumentTokenizer")
                raise ImportError
            self._tokenizer = Tokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer


class LengthCounter(PipelineStep):
    name = "📊 Document length counter"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
    ):
        super().__init__()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for document in data:
            count = document.metadata["token_count"]
            self.stats[count].update(1)
            yield document
