from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.tokenization import PipelineStepWithTokenizer


class TokensCounter(PipelineStepWithTokenizer):
    """Count the number of tokens in each document.
        This pipeline step uses the HuggingFace fast tokenizers library to count the number of tokens in each document.
        It doesn't save the tokenized documents, only the token count.

    Args:
        tokenizer_name_or_path (str): the name or path of the tokenizer to use, from the HuggingFace tokenizers library or a local file.
        count_eos_token (bool): whether to count the EOS token on each document.
    """

    name = "📊 Counter"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        tokenizer_name_or_path: str = "gpt2",  # tokenizer to use, from HF or a local file path
        count_eos_token: bool = False,  # whether to count the EOS token on each document
        overwrite: bool = True,  # re-tokenize and recompute nb of tokens even if they are already in metadata["tokens_count"]
    ):
        """
        Initializes the token counting pipeline step.

        Args:
            tokenizer_name_or_path: Name or path of tokenizer to use (from HF or local).
            count_eos_token: Whether to include the EOS token in the token count per document. (basically +1 per document)
            overwrite: Whether to re-tokenize and recompute the number of tokens even if they are already stored in metadata["tokens_count"]
        """
        super().__init__()
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.count_eos_token = count_eos_token
        self.overwrite = overwrite

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:
          DocumentsPipeline: The pipeline with updated documents, each having a new or updated `token_count` in its metadata.

        """
        for document in data:
            if "token_count" in document.metadata and not self.overwrite:
                count = document.metadata["token_count"]
            else:
                with self.track_time():
                    count = len(self.tokenizer.encode(document.text).ids)
                    if self.count_eos_token:
                        count += 1
                document.metadata["token_count"] = count
            self.stat_update("tokens", value=count)
            yield document


class LengthCounter(PipelineStep):
    """This pipeline step can be used after a TokensCounter or Tokenization step
    to create an histogram of the document token length.

    It doesn't modify the documents, only update a counter for in the stats with each document length.
    Will absolutely spam the hell out of your stats.json
    """

    name = "📊 Document length counter"
    type = "🔢 - TOKENIZER"

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
