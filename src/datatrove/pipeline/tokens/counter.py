from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.tokenization import PipelineStepWithTokenizer, batched


class TokensCounter(PipelineStepWithTokenizer):
    """Count the number of tokens in each document.
        This pipeline step uses the HuggingFace fast tokenizers library to count the number of tokens in each document.
        It doesn't save the tokenized documents, only the token count.

    Args:
        tokenizer_name_or_path (str): the name or path of the tokenizer to use, from the HuggingFace tokenizers library or a local file.
        count_eos_token (bool): whether to count the EOS token on each document. (basically +1 per document)
        batch_size: batch size for tokenization
    """

    name = "📊 Counter"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        tokenizer_name_or_path: str = "gpt2",  # tokenizer to use, from HF or a local file path
        count_eos_token: bool = False,  # whether to count the EOS token on each document
        batch_size: int = 10000,  # batch size for tokenization
    ):
        super().__init__()
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.count_eos_token = count_eos_token
        self.batch_size = batch_size

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:
          DocumentsPipeline: The pipeline with updated documents, each having a new or updated `token_count` in its metadata.

        """
        from tokenizers import Encoding

        # tokenize document's text in batches to go faster
        for batch in batched(data, self.batch_size):
            with self.track_time(unit="batch"):
                encoded_batch: list[Encoding] = self.tokenizer.encode_batch([document.text for document in batch])
            for document, encoded in zip(batch, encoded_batch):
                count = len(encoded.ids)
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
