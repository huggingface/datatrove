import mmap
import struct

from loguru import logger
from numpy.random import default_rng

from datatrove.data import DocumentsPipeline
from datatrove.io import OutputDataFolder, OutputDataFile
from datatrove.pipeline.base import PipelineStep

TOKENIZERS_INSTALLED = True
try:
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing
except ImportError:
    TOKENIZERS_INSTALLED = False


class TokenizedFile:
    def __init__(
            self,
            output_folder: OutputDataFolder,
            filename: str,
            save_index: bool = True,
    ):
        self.output_folder = output_folder
        self.filename = filename
        self.save_index = save_index
        self.tokens_file: OutputDataFile | None = None
        self.write_idx = 0
        self.doc_ends = []

    def __enter__(self):
        self.tokens_file = self.output_folder.get_file(self.filename, lambda x: open(x, "wb"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tokens_file.close()
        # save index: document boundaries
        if self.save_index:
            index_file = self.output_folder.get_file(f"{self.filename}.index", lambda x: open(x, "wb"))
            # save total number of documents
            index_file.file_handler.write(struct.pack('<I', len(self.doc_ends)))
            # save document boundaries
            index_file.file_handler.write(struct.pack('<%sI' % len(self.doc_ends), *self.doc_ends))
            index_file.close()

    def cleanup(self):
        self.doc_ends = []
        self.output_folder.delete_file(self.filename)

    def write_bytes(self, tk_bytes: bytes):
        self.tokens_file.file_handler.write(tk_bytes)
        # 1 token = 2 bytes (uint16)
        self.write_idx += len(tk_bytes) // 2
        # save each document's boundary
        self.doc_ends.append(self.write_idx)

    def write(self, tokens: list[int]):
        # get the bytes for uint16 (H)
        self.write_bytes(struct.pack('<%sH' % len(tokens), *tokens))


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
        self.save_loss_metadata = save_loss_metadata,
        self.shuffle = shuffle
        self._tokenizer = None
        self.rand = default_rng()

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def write_unshuffled(self, data, filename):
        with TokenizedFile(
                self.output_folder,
                filename,
                save_index=not self.shuffle
        ) as unshuff:
            # tokenize each document's text and write its tokens sequentially to the output .ds
            for document in data:
                encoded = self.tokenizer.encode(document.content)
                tokens = encoded.ids
                unshuff.write(tokens)
        return unshuff

    def shuffle_tokens(self, origin: TokenizedFile, destination: str):
        # open original file in read mode
        file = self.output_folder.get_file(origin.filename, lambda x: open(x, "r+b"), overwrite=True)
        with TokenizedFile(
                self.output_folder,
                destination
        ) as shuff:
            # mmap the original file
            with mmap.mmap(file.file_handler.fileno(), 0) as orig_tokens:
                # shuffle doc_id
                for doc_id in self.rand.permutation(len(origin.doc_ends)):
                    # get start and end from the boundaries
                    start, end = origin.doc_ends[doc_id - 1] if doc_id > 0 else 0, origin.doc_ends[doc_id]
                    # copy the bytes. each token is 2 bytes
                    shuff.write_bytes(orig_tokens[start * 2: end * 2])
        file.close()
        origin.cleanup()

    def get_output_filename(self, rank, name):
        return "_".join([x for x in [f"{rank:05d}", self.save_filename, f"{name}.ds"] if x])

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        unshuf_filename = self.get_output_filename(rank, "unshuffled")
        unshuffled_file: TokenizedFile = self.write_unshuffled(data, unshuf_filename)
        if self.shuffle:
            shuffled_filename = self.get_output_filename(rank, "shuffled")
            self.shuffle_tokens(unshuffled_file, shuffled_filename)
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
