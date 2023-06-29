import mmap
import struct

import numpy as np
from loguru import logger
from numpy.random import default_rng

from datatrove.data import DocumentsPipeline, Document
from datatrove.io import OutputDataFolder, OutputDataFile
from datatrove.pipeline.base import PipelineStep

TOKENIZERS_INSTALLED = True
try:
    from tokenizers import Tokenizer, Encoding
    from tokenizers.processors import TemplateProcessing
except ImportError:
    TOKENIZERS_INSTALLED = False


class TokenizedFile:
    def __init__(
            self,
            output_folder: OutputDataFolder,
            filename: str,
            save_index: bool = True,
            save_loss_metadata: bool = True
    ):
        self.output_folder = output_folder
        self.filename = filename
        self.save_index = save_index
        self.save_loss_metadata = save_loss_metadata
        self.tokens_file: OutputDataFile | None = None
        self.loss_file: OutputDataFile | None = None
        self.write_idx = 0
        self.doc_ends = []

    def __enter__(self):
        self.tokens_file = self.output_folder.get_file(self.filename, lambda x: open(x, "wb"))
        if self.save_loss_metadata:
            self.loss_file = self.output_folder.get_file(f"{self.filename}.loss", lambda x: open(x, "wb"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tokens_file:
            self.tokens_file.close()
        if self.loss_file:
            self.loss_file.close()
        # save index: document boundaries
        if self.save_index:
            index_file = self.output_folder.get_file(f"{self.filename}.index", lambda x: open(x, "wb"))
            # save total number of documents
            # index_file.file_handler.write(struct.pack('<I', len(self.doc_ends)))
            # save document boundaries
            index_file.file_handler.write(struct.pack('<%sI' % len(self.doc_ends), *self.doc_ends))
            index_file.close()

    def cleanup(self):
        self.doc_ends = []
        self.output_folder.delete_file(self.filename)
        self.output_folder.delete_file(f"{self.filename}.index")
        self.output_folder.delete_file(f"{self.filename}.loss")

    def write_bytes(self, tk_bytes: bytes):
        self.tokens_file.file_handler.write(tk_bytes)
        # 1 token = 2 bytes (uint16)
        self.write_idx += len(tk_bytes) // 2
        # save each document's boundary
        self.doc_ends.append(self.write_idx)

    def write_loss_bytes(self, l_bytes: bytes):
        if self.save_loss_metadata:
            self.loss_file.file_handler.write(l_bytes)

    def write(self, tokens: list[int], loss_values: np.ndarray | None):
        # get the bytes for uint16 (H)
        self.write_bytes(struct.pack('<%sH' % len(tokens), *tokens))
        if loss_values is not None:
            self.write_loss_bytes(struct.pack('<%s?' % len(loss_values), *loss_values))

    def copy(self, destination: str, ordering: np.ndarray = None):
        # open original file in read mode
        tokens_file = self.output_folder.get_file(self.filename, lambda x: open(x, "r+b"), overwrite=True)
        loss_file = None if not self.loss_file else \
            self.output_folder.get_file(f"{self.filename}.loss", lambda x: open(x, "r+b"), overwrite=True)
        with TokenizedFile(
                self.output_folder,
                destination,
                save_loss_metadata=self.save_loss_metadata
        ) as new_file:
            # mmap the original file
            orig_tokens = mmap.mmap(tokens_file.file_handler.fileno(), 0)
            orig_loss = mmap.mmap(loss_file.file_handler.fileno(), 0) if loss_file else None
            # shuffle doc_id
            for doc_id in ordering:
                # get start and end from the boundaries
                start, end = self.doc_ends[doc_id - 1] if doc_id > 0 else 0, self.doc_ends[doc_id]
                # copy the bytes. each token is 2 bytes
                new_file.write_bytes(orig_tokens[start * 2: end * 2])
                # copy loss values (1 byte per token)
                new_file.write_loss_bytes(orig_loss[start: end])
            # close mmaps
            orig_tokens.close()
            orig_loss.close()
        # close files
        tokens_file.close()
        loss_file.close()
        return new_file


class DocumentTokenizer(PipelineStep):
    def __init__(
            self,
            output_folder: OutputDataFolder,
            save_filename: str = None,  # if defined, the final output filename will be this
            tokenizer_name: str = "gpt2",  # tokenizer to use, from HF
            eos_token: str = "<|endoftext|>",  # whether to add the EOS token after each document
            save_loss_metadata: bool = True,  # save the loss information
            shuffle: bool = True,  # whether to shuffle documents in the dataset
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.output_folder = output_folder
        self.save_filename = save_filename
        self.tokenizer_name = tokenizer_name
        self.eos_token = eos_token
        self.save_loss_metadata = save_loss_metadata
        self.shuffle = shuffle
        self._tokenizer = None
        self.rand = default_rng()

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def get_loss_values(self, document: Document, encoded: Encoding):
        loss_values = None
        if self.save_loss_metadata:
            loss_values = np.ones((len(encoded.ids)))
            if no_loss := document.metadata.get('no_loss_ranges', None):
                for start, end in no_loss:
                    t_start, t_end = encoded.char_to_token(start), encoded.char_to_token(end)
                    # set loss to 0
                    loss_values[t_start:t_end] = 0
                    if t_end is None or t_end >= len(encoded.ids):
                        # drop this last section
                        loss_values = loss_values[:t_start]
        return loss_values

    def write_unshuffled(self, data, filename):
        with TokenizedFile(
                self.output_folder,
                filename,
                save_index=not self.shuffle,
                save_loss_metadata=self.save_loss_metadata
        ) as unshuff:
            # tokenize each document's text and write its tokens sequentially to the output .ds
            for document in data:
                encoded: Encoding = self.tokenizer.encode(document.content)
                tokens = encoded.ids
                # loss values
                loss_values = self.get_loss_values(document, encoded)
                if loss_values is not None and len(loss_values) < len(tokens):
                    tokens = tokens[:len(loss_values)]
                # write bytes to disk
                unshuff.write(tokens, loss_values)
        return unshuff

    def get_output_filename(self, rank, name):
        return "_".join([x for x in [f"{rank:05d}", self.save_filename, f"{name}.ds"] if x])

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        unshuf_filename = self.get_output_filename(rank, "unshuffled")
        unshuffled_file: TokenizedFile = self.write_unshuffled(data, unshuf_filename)
        if self.shuffle:
            shuffled_filename = self.get_output_filename(rank, "shuffled")
            unshuffled_file.copy(shuffled_filename, self.rand.permutation(len(unshuffled_file.doc_ends)))
            unshuffled_file.cleanup()
        self.output_folder.close()

    @property
    def tokenizer(self):
        if not self._tokenizer:
            if not TOKENIZERS_INSTALLED:
                logger.error("FastText is required to run LanguageFilter")
                raise ImportError
            self._tokenizer = Tokenizer.from_pretrained(self.tokenizer_name)
            if self.eos_token:
                self._tokenizer.post_processor = TemplateProcessing(
                    single="$A <EOS>",
                    special_tokens=[("<EOS>", self.tokenizer.token_to_id(self.eos_token))],
                    pair=None
                )
        return self._tokenizer
