import itertools
import mmap
import struct

import humanize
import numpy as np
from loguru import logger
from numpy.random import default_rng

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import BaseOutputDataFile, BaseOutputDataFolder
from datatrove.pipeline.base import PipelineStep


TOKENIZERS_INSTALLED = True
try:
    from tokenizers import Encoding, Tokenizer
    from tokenizers.processors import TemplateProcessing
except ImportError:
    TOKENIZERS_INSTALLED = False


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
        output_folder: BaseOutputDataFolder,
        filename: str,
        save_index: bool = True,
        save_loss_metadata: bool = False,
    ):
        self.output_folder = output_folder
        self.filename = filename
        self.save_index = save_index
        self.save_loss_metadata = save_loss_metadata
        self.write_idx = 0
        self.doc_ends = []

        self.tokens_file = self.output_folder.open(self.filename, mode="wb")
        self.loss_file: BaseOutputDataFile | None = None
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
        self.tokens_file.delete()
        if self.loss_file:
            self.loss_file.delete()

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

    def copy(self, destination: str, ordering: np.ndarray = None):
        # open original file in read mode
        tokens_file = self.output_folder.open(self.filename, mode="r+b")
        loss_file = None if not self.loss_file else self.output_folder.open(f"{self.filename}.loss", mode="r+b")
        new_file = TokenizedFile(self.output_folder, destination, save_loss_metadata=self.save_loss_metadata)
        # mmap the original file
        orig_tokens = mmap.mmap(tokens_file._file_handler.fileno(), 0)
        orig_loss = mmap.mmap(loss_file._file_handler.fileno(), 0) if loss_file else None
        # shuffle doc_id
        for doc_id in ordering:
            # get start and end from the boundaries
            start, end = self.doc_ends[doc_id - 1] if doc_id > 0 else 0, self.doc_ends[doc_id]
            # copy the bytes. each token is 2 bytes
            new_file.write_bytes(orig_tokens[start * 2 : end * 2])
            # copy loss values (1 byte per token)
            if orig_loss:
                new_file.write_loss_bytes(orig_loss[start:end])
        # close mmaps
        orig_tokens.close()
        if orig_loss:
            orig_loss.close()
        return new_file

    def save_final_metadata(self, tokenizer_name: str | None = None, token_count: int = -1, filename: str = None):
        if not tokenizer_name:
            tokenizer_name = "Unknown Tokenizer"
        if filename is None:
            filename = self.filename
        with self.output_folder.open(f"{filename}.metadata") as f:
            if token_count == -1:
                token_count = self.write_idx
            f.write("\n".join([tokenizer_name, str(token_count), humanize.metric(token_count, unit="T")]))


class DocumentTokenizer(PipelineStep):
    name = "✍️ Writer"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        output_folder: BaseOutputDataFolder,
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
        self.output_folder = output_folder
        self.save_filename = save_filename
        self.tokenizer_name = tokenizer_name
        self.eos_token = eos_token
        self.save_loss_metadata = save_loss_metadata
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._tokenizer = None
        self.rand = default_rng(seed)
        self.save_final_metadata = save_final_metadata

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def get_loss_values(self, document: Document, encoded: Encoding):
        loss_values = None
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
        unshuff = TokenizedFile(
            self.output_folder, filename, save_index=not self.shuffle, save_loss_metadata=self.save_loss_metadata
        )
        # tokenize document's text in batches to go faster – we compute loss values independently if needed
        for batch in batched(data, self.batch_size):
            with self.track_time():
                encoded_batch: Encoding = self.tokenizer.encode_batch([document.content for document in batch])
                loss_values_batch = [
                    self.get_loss_values(document, encoded) for document, encoded in zip(batch, encoded_batch)
                ]
            for encoded, loss_values in zip(encoded_batch, loss_values_batch):
                with self.track_time():
                    tokens = encoded.ids
                    if loss_values is not None and len(loss_values) < len(tokens):
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
            self.output_folder.close()
            logger.warning("No data saved.")
            return
        if self.shuffle:
            shuffled_filename = self.get_output_filename(rank, "shuffled")
            # get new TokenizedFile, shuffling docs from original one
            new_outputfile = outputfile.copy(shuffled_filename, self.rand.permutation(len(outputfile.doc_ends)))
            # remove and replace original file
            outputfile.cleanup()
            outputfile = new_outputfile
        outputfile.close()
        if self.save_final_metadata:
            outputfile.save_final_metadata(self.tokenizer_name)
        self.output_folder.close()

    @property
    def tokenizer(self) -> Tokenizer:
        if not self._tokenizer:
            if not TOKENIZERS_INSTALLED:
                logger.error("`tokenizers` is required to run DocumentTokenizer")
                raise ImportError
            self._tokenizer = Tokenizer.from_pretrained(self.tokenizer_name)
            if self.eos_token:
                self._tokenizer.post_processor = TemplateProcessing(
                    single="$A <EOS>",
                    special_tokens=[("<EOS>", self.tokenizer.token_to_id(self.eos_token))],
                    pair=None,
                )
        return self._tokenizer
