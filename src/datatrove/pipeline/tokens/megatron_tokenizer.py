from tqdm import tqdm
import struct
import orjson
import logging
import smart_open
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Optional, NewType, Callable
from concurrent.futures import ThreadPoolExecutor

import humanize
import subprocess
import numpy as np
import pandas as pd
from loguru import logger
from numpy.random import default_rng
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer
from tokenizers import Encoding, Tokenizer

from datatrove.utils.batching import batched
from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolder, DataFolderLike, get_datafolder
from datatrove.utils.tokenization import PipelineStepWithTokenizer
from datatrove.utils.indexed_dataset import _IndexWriter, _IndexReader, DType


SHUFFLING_READ_BLOCK_SIZE = 50000  # read 50kb at a time only (~mean + 2sigmas for final filtered common crawl docs)
# at a time to avoid reading a lot of data into cache and then not using it when jumping again
SHUFFLING_CACHE_TYPE = "none"  # do not cache as we are only jumping around and not reading sequentially

if TYPE_CHECKING:
    from tokenizers import Encoding, Tokenizer


TokenizerKind = NewType("TokenizerKind", Tokenizer | PreTrainedTokenizerFast | PreTrainedTokenizer)


def import_tokenizer_class(path):
    """ Import tokemizer implementation from custom python script.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "tokenizer.py")
    if not os.path.exists(path):
        raise OSError(f"File {path} not exists.")
    module_name = "tokenizer"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Tokenizer


class Incrementer:
    """ An incrementer to generate integer document ID.
    """

    def __init__(self, offset, step):
        self.offset = offset
        self.step = step
        self._counter = 0

    def next(self):
        ret = self._counter + self.offset
        self._counter += self.step
        return ret


class TokenizerTypes(StrEnum):
    HuggingFace = "HuggingFace"
    TikToken = "TikToken"
    Tokenizer = "Tokenizer"


def load_tokenizer(model: str | Callable, vocab: str | None=None, type: str | TokenizerTypes=TokenizerTypes.HuggingFace) -> TokenizerKind:
    """ Load various type of tokenizers.
    """
    logger.info(f"Using tokenizer: {type!s}.")
    if type == TokenizerTypes.HuggingFace:
        return AutoTokenizer.from_pretrained(model)
    elif type == TokenizerTypes.TikToken:
        if vocab is not None and (os.path.isdir(model) or model.endswith(".py")):
            return import_tokenizer_class(model)(vocab)
        else:
            raise NotImplementedError
    elif type == TokenizerTypes.Tokenizer:
        if os.path.exists(name_or_path):
            return Tokenizer.from_file(name_or_path)
        return Tokenizer.from_pretrained(name_or_path)
    elif callable(model):
        return model()
    else:
        raise TypeError("Unable to load tokenizer.")


def encode_batch(tokenizer: TokenizerKind, batched_data, num_process: Optional[int]=8) -> list[list[int]]:
    """ Compatibility layer for batched encoding.
    """
    if hasattr(tokenizer, "encode_batch"):
        result = tokenizer.encode_batch(batched_data)
        if isinstance(result, Encoding):
            result = result.ids
        return result
    with ThreadPoolExecutor(min(num_process, len(batched_data))) as e:
        return list(e.map(tokenizer.encode, batched_data))


def write_index(index_file, dtype, doc_ends, doc_indices: list[int] | None = None):
    """ Write sequence and document indices into Megatron compatible index files.
    """
    with _IndexWriter(index_file, dtype) as writer:
        sequence_lengths = np.array(doc_ends, dtype=int)
        if len(sequence_lengths) > 1:
            sequence_lengths[1:] -= doc_ends[:-1]
        sequence_lengths = sequence_lengths.tolist()
        if doc_indices is None:
            document_indices = np.arange(len(sequence_lengths), dtype=int).tolist()
        else:
            document_indices = np.array(doc_indices, dtype=int).tolist()
        document_indices.append(len(document_indices))
        writer.write(sequence_lengths, None, document_indices)


def read_index(index_file):
    """ Read sequence and document indices from Megatron compatible index files.
    """
    return _IndexReader(index_file, False)


def load_doc_ends_indices(index_file):
    """Load doc_ends and doc_indices from idx file."""
    index_file = read_index(index_file)
    # doc_end = sequence start + sequence length
    doc_ends = index_file.sequence_pointers // index_file.dtype_size + index_file.sequence_lengths
    doc_indices = index_file.document_indices.copy()
    return doc_ends, doc_indices


class TokenizedFile:
    """Class to write tokenized documents to local/remote folders.
        Handles writing the tokenized document, an index file with the document ends (in tokens).
        Also handles shuffling the documents inside the file at the end and providing a new TokenizedFile instance with the new ordering.

    Args:
        output_folder (DataFolderLike): the output folder where to save the tokenized documents
        filename (str): the filename to use
        save_index (bool): whether to save the index file (document boundaries)
        upload_block_size (int): the fsspec size of the upload block for remote filesystems (S3)
        token_size (int): size of each token, in bytes

    """

    def __init__(
        self,
        output_folder: DataFolderLike,
        filename: str,
        save_index: bool = True,
        save_doc_meta: bool = True,
        upload_block_size: int | None = None,
        tokenizer_name_or_path: str | None = None,
        save_final_metadata: bool = False,
        token_size: int = 4,
        suffix: str = ".npy",
        compress: bool = False,
        compression_cmd: str = None,
    ):
        self.output_folder = get_datafolder(output_folder)
        self.filename = filename
        self.save_index = save_index
        self.save_doc_meta = save_doc_meta
        self.upload_block_size = upload_block_size
        self.write_idx = 0
        self.token_size = token_size
        self.token_format = "I" if self.token_size == 4 else "H"
        self.doc_ends = []
        self.doc_indices = []
        self.doc_meta = []
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.save_final_metadata = save_final_metadata

        self.suffix = suffix
        self.tokens_file = self.output_folder.open(self.filename, mode="wb", block_size=upload_block_size)
        self.compress = compress
        self.compression_cmd = compression_cmd if compression_cmd is not None else "gzip %s"

    @property
    def token_dtype(self):
        return np.uint16 if self.token_size <= 2 else np.uint32

    def __len__(self):
        return self.doc_ends[-1] if self.doc_ends else 0

    def close(self):
        """Close the files and save the index."""

        logger.info("Close tokens file.")
        if self.tokens_file:
            self.tokens_file.close()
            if self.compress:
              logger.info("Compress tokens file.")
              subprocess.check_call(self.compression_cmd % self.output_folder._join(self.filename), shell=True)

        # save index: document boundaries
        logger.info("Save idx file.")
        if self.save_index:
            write_index(
                self.get_idx_file(),
                self.token_dtype,
                self.doc_ends,
                doc_indices=self.doc_indices,
                )

        logger.info("Save dataset metadata.")
        if self.save_final_metadata:
            self.write_final_metadata()

        logger.info("Save per-document metadata.")
        if self.save_doc_meta and self.doc_meta:
            with smart_open.open(self.get_doc_meta_file(), "wt") as f:
                for doc in self.doc_meta:
                    f.write(orjson.dumps(doc, option=orjson.OPT_APPEND_NEWLINE).decode('utf8'))
            #  df = pd.DataFrame(self.doc_meta)
            #  df.to_json(self.get_doc_meta_file(), lines=True, orient='records')

    def get_idx_file(self):
      return self.output_folder._join(self.filename.replace(self.suffix, ".idx"))

    def get_doc_meta_file(self):
      return self.output_folder._join(self.filename.replace(self.suffix, ".docmeta.jsonl.gz"))

    def load_doc_ends(self):
        """Load doc_ends and doc_indices from idx file."""
        self.doc_ends, self.doc_indices = load_doc_ends_indices(self.get_idx_file())

    def load_doc_meta(self):
        self.doc_meta = pd.read_json(self.get_doc_meta_file(), lines=True)

    def cleanup(self):
        """Remove the files and the index."""
        self.doc_ends = []
        self.doc_indices = []
        self.output_folder.rm_file(self.filename)
        if self.save_final_metadata and self.output_folder.exists(f"{self.filename}.metadata"):
            self.output_folder.rm_file(f"{self.filename}.metadata")

    def write_bytes(self, tk_bytes: bytes, doc_ends: list[int] = None):
        """Write tk_bytes to the tokens file and update the document boundaries with a new document end (in tokens).

        Args:
          tk_bytes: bytes:
          doc_ends: list[int]  (Default value = None): optional list of document ends (in tokens) if writing several documents at once

        Returns:
        """
        self.tokens_file.write(tk_bytes)
        if doc_ends is not None:
            # We've written several documents at once
            self.doc_ends.extend([d + self.write_idx for d in doc_ends])
            self.write_idx += len(tk_bytes) // self.token_size
        else:
            # We've written a single document
            self.write_idx += len(tk_bytes) // self.token_size
            # save each document's boundary
            self.doc_ends.append(self.write_idx)

    def write(self, tokens: list[int], index: int, doc_meta: dict = None):
        """Write tokens to the files.

        Args:
            tokens (list[int]): the tokens to write
        """
        # get the bytes
        self.write_bytes(struct.pack(f"<%s{self.token_format}" % len(tokens), *tokens))
        self.doc_indices.append(index) 
        if self.doc_meta is not None:
            self.doc_meta.append(doc_meta)

    def copy(
        self,
        save_filename: str,
        ordering: np.ndarray,
        new_output_folder: DataFolder = None,
        rank: int = 0,
        max_tokens_per_file: int = None,
    ) -> "TokenizedFile":
        """Close the current tokenized file and copy its content to a new file, shuffling the document order with provided ordering.

        Args:
            save_filename (str): the new filename in new_output_folder
            ordering (np.ndarray): the new ordering of the documents
            new_output_folder (DataFolder): the new output folder to use
            rank: used to get filename
            max_tokens_per_file: split into small chunk files each with max this number of tokens
        Returns:
            TokenizedFile: the new tokenized file
        """
        # open original file in read mode
        with self.output_folder.open(
            self.filename, mode="rb", cache_type=SHUFFLING_CACHE_TYPE, block_size=SHUFFLING_READ_BLOCK_SIZE
        ) as tokens_file:
            sub_rank = 0
            destination = get_output_filename(save_filename, rank, "shuffled", sub_rank)

            new_file = TokenizedFile(
                self.output_folder if not new_output_folder else new_output_folder,
                destination,
                upload_block_size=self.upload_block_size,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                save_final_metadata=self.save_final_metadata,
                token_size=self.token_size,
            )
            logger.info(f"Shuffling in {destination}...")
            # shuffle doc_id
            total_tokens_written = 0
            for doc_id in tqdm(ordering, desc=f"Shuffling documents", unit="documents"):
                # get start and end from the boundaries
                start, end = self.doc_ends[doc_id - 1] if doc_id > 0 else 0, self.doc_ends[doc_id]
                # copy the bytes. each token is token_size bytes
                tokens_file.seek(start * self.token_size)
                new_file.write_bytes(tokens_file.read((end - start) * self.token_size))
                #
                new_file.doc_indices.append(self.doc_indices[doc_id])
                new_file.doc_meta.append(self.doc_meta[doc_id])
                #
                total_tokens_written += end - start
                if max_tokens_per_file and total_tokens_written > max_tokens_per_file:
                    new_file.close()
                    sub_rank += 1
                    destination = get_output_filename(save_filename, rank, "shuffled", sub_rank)
                    new_file = TokenizedFile(
                        self.output_folder if not new_output_folder else new_output_folder,
                        destination,
                        upload_block_size=self.upload_block_size,
                        tokenizer_name_or_path=self.tokenizer_name_or_path,
                        save_final_metadata=self.save_final_metadata,
                        token_size=self.token_size,
                    )
                    logger.info(f"Shuffling in {destination}...")
                    total_tokens_written = 0
            #  new_file.doc_indices = np.array(self.doc_indices)[ordering]
            new_file.close()
            return new_file

    def write_final_metadata(self, token_count: int = -1, filename: str = None):
        """Save the final metadata file with the tokenizer name and the token count.

        Args:
            tokenizer_name (str | None): the tokenizer name to save (Default value = None)
            token_count (int): the token count to save (Default value = -1)
            filename: str:  (Default value = None)
        """
        tokenizer_name = self.tokenizer_name_or_path
        if not tokenizer_name:
            tokenizer_name = "Unknown Tokenizer" + "|" + str(self.token_size)
        if filename is None:
            filename = self.filename
        with self.output_folder.open(f"{filename}.metadata", "wt") as f:
            if token_count == -1:
                token_count = self.write_idx
            f.write(
                "\n".join(
                    [
                        tokenizer_name + "|" + str(self.token_size),
                        str(token_count),
                        humanize.metric(token_count, unit="T"),
                    ]
                )
            )
        with self.output_folder.open(f"{filename}.datalist", "wt") as f:
            if token_count == -1:
                token_count = self.write_idx
            prefix = self.output_folder._join(self.filename).replace(self.suffix, "")
            f.write(f"{token_count:<12d} {prefix}\n")


def get_output_filename(save_filename, rank: int, name: str, sub_rank: int = None, suffix: str=".npy"):
    """Get an output filename for the rank and a sub-step name (unshuffled/shuffled)."""
    if sub_rank is not None:
        return "_".join([x for x in [save_filename, f"{rank:05d}", f"{sub_rank:05d}", f"{name}{suffix}"] if x])
    return "_".join([x for x in [save_filename, f"{rank:05d}", f"{name}{suffix}"] if x])


class MegatronTokenizer(PipelineStepWithTokenizer):
    """Tokenize the documents in the pipeline using the HuggingFace fast tokenizers library.
        This pipeline step saves the tokenized documents locally/remotely in a set of files and optionally shuffles documents inside each file.

        You can use a DocumentTokenizerMerger to merge the tokenized files into a single file while also shuffling the file order.

    Args:
        output_folder (DataFolderLike): the output folder where to save the tokenized documents
        local_working_dir (str | None): a local working directory to use for temporary files (before internal shuffling)
            if None we shuffle in output_folder (can be very slow if it's a remote location)
        save_filename (str): the filename to use for the final tokenized files (default: None – use the default filename)
        tokenizer_name_or_path (str): the name or path of the tokenizer to use, from the HuggingFace tokenizers library (default: "gpt2")
        eos_token (str): whether to add the EOS token after each document (default: "<|endoftext|>")
        shuffle (bool): whether to shuffle documents in the dataset (default: True)
        batch_size (int): batch size for tokenization (default: 1000)
        seed (int): the seed to use for shuffling
        save_final_metadata (bool): whether to save the final metadata (default: True)
        upload_block_size (int | None): the fsspec size of the upload block for remote filesystems (S3)
            You can set this if your s3 uploads are failing because of "Part number must be an integer between 1 and 10000, inclusive".
            Example: 20 * 2**20 (20MB)
    """

    name = "✍️ Megatron Writer"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        output_folder: DataFolderLike,
        local_working_dir: DataFolderLike | None = None,
        save_filename: str = None,  # if defined, the final output filename will be this
        tokenizer_name_or_path: str = "gpt2",  # tokenizer to use, from HF or a local
        tokenizer_vocab: str | None = None,
        tokenizer_type: str | TokenizerTypes = TokenizerTypes.HuggingFace,
        token_size: int = 4,
        # eos_token: str = "<|endoftext|>",  # whether to add the EOS token after each document
        shuffle: bool = True,  # whether to shuffle documents in the dataset,
        batch_size: int = 10000,  # batch size for tokenization
        max_tokens_per_file: int = None,  # max tokens per file to get more (smaller) shuffled output files
        seed: int = None,
        save_final_metadata: bool = True,
        upload_block_size: int | None = None,
        # you can set this if your s3 uploads are failing because of "Part
        # number must be an integer between 1 and 10000, inclusive". Example: 20 * 2**20 (20MB)
        suffix: str = ".npy",
        compress: bool = False,
        compression_cmd: str = None,
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.local_working_dir = get_datafolder(local_working_dir) if local_working_dir else None
        if self.local_working_dir and not self.local_working_dir.is_local():
            raise ValueError("local_working_dir must be a local path")
        if self.local_working_dir is None and shuffle and not self.output_folder.is_local():
            logger.warning(
                "local_working_dir is not set and output folder is not local. This may slow down the process."
            )
        self.save_filename = save_filename
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer_vocab = tokenizer_vocab
        self.tokenizer_type = tokenizer_type
        # self.eos_token = eos_token
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.rand = default_rng(seed)
        self.save_final_metadata = save_final_metadata
        self.upload_block_size = upload_block_size
        self.max_tokens_per_file = max_tokens_per_file
        self.suffix = suffix
        self._token_size = token_size
        self.compress = compress
        self.compression_cmd = compression_cmd

    @property
    def token_dtype(self):
        return np.uint16 if self.token_size <= 2 else np.uint32

    @property
    def tokenizer(self) -> TokenizerKind:
        if not self._tokenizer:
            if not self.tokenizer_name_or_path:
                raise ValueError("self.tokenizer_name_or_path needs to be set!")
            self._tokenizer = load_tokenizer(self.tokenizer_name_or_path, self.tokenizer_vocab, self.tokenizer_type)
            if self._post_processor:
                self._tokenizer.post_processor = self._post_processor
            elif self.eos_token:
                self._tokenizer.post_processor = TemplateProcessing(
                    single="$A <EOS>",
                    special_tokens=[("<EOS>", self.tokenizer.token_to_id(self.eos_token))],
                    pair=None,
                )
        return self._tokenizer

    def write_unshuffled(self, data: DocumentsPipeline, filename: str, rank: int = 0, world_size: int = 1):
        """Tokenize documents with the tokenizer in batches and write the unshuffled tokenized documents to a file.

        Args:
            data (DocumentsPipeline): the documents to process
            filename (str): the filename to use for the output file
        """
        from tokenizers import Encoding

        unshuff = TokenizedFile(
            self.output_folder if not self.shuffle or not self.local_working_dir else self.local_working_dir,
            filename,
            save_index=not self.shuffle,
            upload_block_size=self.upload_block_size,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            save_final_metadata=self.save_final_metadata,
            token_size=self.token_size,
            compress=self.compress,
            compression_cmd=self.compression_cmd,
        )
        inc = Incrementer(rank, world_size)
        # tokenize document's text in batches to go faster
        for batch in tqdm(batched(data, self.batch_size), desc="Writing unshuffled documents", unit="batches"):
            with self.track_time(unit="batch"):
                encoded_batch: list[Encoding] = encode_batch(self.tokenizer, [document.text for document in batch])
                for document, tokens in zip(batch, encoded_batch):
                    # save metadata for tracing purpose
                    doc_meta = {
                        "source_file": document.metadata["_source_file"],
                        "id_in_file": document.metadata["_id_in_file"],
                    }
                    # write bytes to disk
                    unshuff.write(tokens, inc.next(), doc_meta)
                    # save stats
                    self.stat_update("tokens", value=len(tokens))
        unshuff.close()
        return unshuff

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Main method to run the tokenization.
            We first batch tokenize the documents and write them to a file.
            Then we shuffle the documents and write them to a new file if self.shuffle is True (and remove the original file)

        Args:
            data: DocumentsPipeline
                The data to be processed as a Generator typically created by a Reader initial pipeline step
            rank: int
                The rank of the process
            world_size: int
                The total number of processes
        """
        unshuf_filename = get_output_filename(self.save_filename, rank, "unshuffled", suffix=self.suffix)
        logger.info(f'Tokenizing in "{unshuf_filename}"...')
        outputfile: TokenizedFile = self.write_unshuffled(data, unshuf_filename, rank=rank, world_size=world_size)
        if len(outputfile) == 0:
            logger.warning("No data saved.")
            return
        if self.shuffle:
            logger.info("Shuffling...")
            # get new TokenizedFile, shuffling docs from original one
            outputfile.copy(
                self.save_filename,
                self.rand.permutation(len(outputfile.doc_ends)),
                self.output_folder,
                max_tokens_per_file=self.max_tokens_per_file,
                rank=rank,
            )
            # remove and replace original file
            outputfile.cleanup()
