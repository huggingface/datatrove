import itertools
import struct
from typing import TYPE_CHECKING

import humanize
import numpy as np
from fsspec.implementations.local import LocalFileSystem
from loguru import logger
from numpy.random import default_rng

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolder, DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep


SHUFFLING_READ_BLOCK_SIZE = 50000  # read 50kb at a time only (~mean + 2sigmas for final filtered common crawl docs)
# at a time to avoid reading a lot of data into cache and then not using it when jumping again
SHUFFLING_CACHE_TYPE = "none"  # do not cache as we are only jumping around and not reading sequentially

if TYPE_CHECKING:
    from tokenizers import Encoding, Tokenizer


def batched(iterable, n):
    """In python 3.12+ we could use itertools.batched instead

    One difference with itertools.batched: we return a list instead of a tuple
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


class TokenizedFile:
    def __init__(
        self,
        output_folder: DataFolderLike,
        filename: str,
        save_index: bool = True,
        save_loss_metadata: bool = False,
    ):
        self.output_folder = get_datafolder(output_folder)
        self.filename = filename
        self.save_index = save_index
        self.save_loss_metadata = save_loss_metadata
        self.write_idx = 0
        self.doc_ends = []

        self.tokens_file = self.output_folder.open(self.filename, mode="wb")
        self.loss_file: DataFolderLike | None = None
        if self.save_loss_metadata:
            self.loss_file = self.output_folder.open(f"{self.filename}.loss", mode="wb")

    def __len__(self):
        return self.doc_ends[-1] if self.doc_ends else 0

    def close(self):
        if self.tokens_file:
            self.tokens_file.close()
        if self.loss_file:
            self.loss_file.close()
        # save index: document boundaries
        if self.save_index:
            index_file = self.output_folder.open(f"{self.filename}.index", mode="wb")
            # save total number of documents
            # index_file.file_handler.write(struct.pack('<I', len(self.doc_ends)))
            # save document boundaries - uint64
            index_file.write(struct.pack("<%sQ" % len(self.doc_ends), *self.doc_ends))
            index_file.close()

    def cleanup(self):
        self.doc_ends = []
        self.output_folder.rm_file(self.filename)
        if self.loss_file:
            self.output_folder.rm_file(f"{self.filename}.loss")

    def write_bytes(self, tk_bytes: bytes):
        self.tokens_file.write(tk_bytes)
        # 1 token = 2 bytes (uint16)
        self.write_idx += len(tk_bytes) // 2
        # save each document's boundary
        self.doc_ends.append(self.write_idx)

    def write_loss_bytes(self, l_bytes: bytes):
        if self.save_loss_metadata:
            self.loss_file.write(l_bytes)

    def write(self, tokens: list[int], loss_values: np.ndarray | None):
        # get the bytes for uint16 (H)
        self.write_bytes(struct.pack("<%sH" % len(tokens), *tokens))
        if loss_values is not None:
            self.write_loss_bytes(struct.pack("<%s?" % len(loss_values), *loss_values))

    def copy(self, destination: str, ordering: np.ndarray = None, new_output_folder: DataFolder = None):
        # open original file in read mode
        self.close()
        with self.output_folder.open(
            self.filename, mode="rb", cache_type=SHUFFLING_CACHE_TYPE, block_size=SHUFFLING_READ_BLOCK_SIZE
        ) as tokens_file:
            loss_file = (
                None
                if not self.loss_file
                else self.output_folder.open(
                    f"{self.filename}.loss",
                    mode="rb",
                    cache_type=SHUFFLING_CACHE_TYPE,
                    block_size=SHUFFLING_READ_BLOCK_SIZE // 2,  # this one is half the size
                )
            )
            new_file = TokenizedFile(
                self.output_folder if not new_output_folder else new_output_folder,
                destination,
                save_loss_metadata=self.save_loss_metadata,
            )
            # shuffle doc_id
            for doc_id in ordering:
                # get start and end from the boundaries
                start, end = self.doc_ends[doc_id - 1] if doc_id > 0 else 0, self.doc_ends[doc_id]
                # copy the bytes. each token is 2 bytes
                tokens_file.seek(start * 2)
                new_file.write_bytes(tokens_file.read((end - start) * 2))
                # copy loss values (1 byte per token)
                if loss_file:
                    loss_file.seek(start)
                    new_file.write_loss_bytes(loss_file.read(end - start))
            if loss_file:
                loss_file.close()
            return new_file

    def save_final_metadata(self, tokenizer_name: str | None = None, token_count: int = -1, filename: str = None):
        if not tokenizer_name:
            tokenizer_name = "Unknown Tokenizer"
        if filename is None:
            filename = self.filename
        with self.output_folder.open(f"{filename}.metadata", "wt") as f:
            if token_count == -1:
                token_count = self.write_idx
            f.write("\n".join([tokenizer_name, str(token_count), humanize.metric(token_count, unit="T")]))


class DocumentTokenizer(PipelineStep):
    name = "✍️ Writer"
    type = "🔢 - TOKENIZER"
    _requires_dependencies = ["tokenizers"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        local_working_dir: str | None = None,
        save_filename: str = None,  # if defined, the final output filename will be this
        tokenizer_name: str = "gpt2",  # tokenizer to use, from HF
        eos_token: str = "<|endoftext|>",  # whether to add the EOS token after each document
        save_loss_metadata: bool = False,  # save the loss information
        shuffle: bool = True,  # whether to shuffle documents in the dataset,
        batch_size: int = 1000,  # batch size for tokenization
        seed: int = None,
        save_final_metadata: bool = True,
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.local_working_dir = get_datafolder(local_working_dir) if local_working_dir else None
        if self.local_working_dir and not isinstance(self.local_working_dir, LocalFileSystem):
            raise ValueError("local_working_dir must be a local path")
        self.save_filename = save_filename
        self.tokenizer_name = tokenizer_name
        self.eos_token = eos_token
        self.save_loss_metadata = save_loss_metadata
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._tokenizer = None
        self.rand = default_rng(seed)
        self.save_final_metadata = save_final_metadata

    def get_loss_values(self, document: Document, encoded: "Encoding"):
        if self.save_loss_metadata:
            loss_values = np.ones((len(encoded.ids)))
            if no_loss := document.metadata.get("no_loss_ranges", None):
                for start, end in no_loss:
                    t_start, t_end = encoded.char_to_token(start), encoded.char_to_token(end)
                    # set loss to 0
                    loss_values[t_start:t_end] = 0
                    if t_end is None or t_end >= len(encoded.ids):
                        # drop this last section
                        loss_values = loss_values[:t_start]
            return loss_values

    def write_unshuffled(self, data: DocumentsPipeline, filename: str):
        from tokenizers import Encoding

        unshuff = TokenizedFile(
            self.output_folder if not self.shuffle or not self.local_working_dir else self.local_working_dir,
            filename,
            save_index=not self.shuffle,
            save_loss_metadata=self.save_loss_metadata,
        )
        # tokenize document's text in batches to go faster – we compute loss values independently if needed
        for batch in batched(data, self.batch_size):
            with self.track_time(unit="batch"):
                encoded_batch: list[Encoding] = self.tokenizer.encode_batch([document.text for document in batch])
                for document, encoded in zip(batch, encoded_batch):
                    tokens = encoded.ids
                    loss_values = self.get_loss_values(document, encoded)
                    if loss_values is not None and len(loss_values) < len(tokens):
                        # crop final section without loss
                        tokens = tokens[: len(loss_values)]
                    # write bytes to disk
                    unshuff.write(tokens, loss_values)
                    # save stats
                    self.stat_update("tokens", value=len(tokens))
        return unshuff

    def get_output_filename(self, rank, name):
        return "_".join([x for x in [self.save_filename, f"{rank:05d}", f"{name}.ds"] if x])

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        unshuf_filename = self.get_output_filename(rank, "unshuffled")
        logger.info(f'Tokenizing in "{unshuf_filename}"...')
        outputfile: TokenizedFile = self.write_unshuffled(data, unshuf_filename)
        if len(outputfile) == 0:
            logger.warning("No data saved.")
            return
        if self.shuffle:
            shuffled_filename = self.get_output_filename(rank, "shuffled")
            # get new TokenizedFile, shuffling docs from original one
            new_outputfile = outputfile.copy(
                shuffled_filename, self.rand.permutation(len(outputfile.doc_ends)), self.output_folder
            )
            # remove and replace original file
            outputfile.cleanup()
            outputfile = new_outputfile
        outputfile.close()
        if self.save_final_metadata:
            outputfile.save_final_metadata(self.tokenizer_name)

    @property
    def tokenizer(self) -> "Tokenizer":
        if not self._tokenizer:
            from tokenizers import Tokenizer
            from tokenizers.processors import TemplateProcessing

            self._tokenizer = Tokenizer.from_pretrained(self.tokenizer_name)
            if self.eos_token:
                self._tokenizer.post_processor = TemplateProcessing(
                    single="$A <EOS>",
                    special_tokens=[("<EOS>", self.tokenizer.token_to_id(self.eos_token))],
                    pair=None,
                )
        return self._tokenizer
