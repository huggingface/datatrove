import struct
from typing import TYPE_CHECKING

import humanize
import numpy as np
from numpy.random import default_rng

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolder, DataFolderLike, get_datafolder
from datatrove.utils.logging import logger
from datatrove.utils.tokenization import PipelineStepWithTokenizer, batched


SHUFFLING_READ_BLOCK_SIZE = 50000  # read 50kb at a time only (~mean + 2sigmas for final filtered common crawl docs)
# at a time to avoid reading a lot of data into cache and then not using it when jumping again
SHUFFLING_CACHE_TYPE = "none"  # do not cache as we are only jumping around and not reading sequentially

if TYPE_CHECKING:
    from tokenizers import Encoding


class TokenizedFile:
    """Class to write tokenized documents to local/remote folders.
        Handles writing the tokenized document, an index file with the document ends (in tokens), and optionally a loss file with loss masks.
        Also handles shuffling the documents inside the file at the end and providing a new TokenizedFile instance with the new ordering.

    Args:
        output_folder (DataFolderLike): the output folder where to save the tokenized documents
        filename (str): the filename to use
        save_index (bool): whether to save the index file (document boundaries)
        save_loss_metadata (bool): whether to save the loss metadata (to mask some tokens during training)
        upload_block_size (int): the fsspec size of the upload block for remote filesystems (S3)
        token_size (int): size of each token, in bytes

    """

    def __init__(
        self,
        output_folder: DataFolderLike,
        filename: str,
        save_index: bool = True,
        save_loss_metadata: bool = False,
        upload_block_size: int | None = None,
        tokenizer_name_or_path: str | None = None,
        save_final_metadata: bool = False,
        token_size: int = 2,
    ):
        self.output_folder = get_datafolder(output_folder)
        self.filename = filename
        self.save_index = save_index
        self.save_loss_metadata = save_loss_metadata
        self.upload_block_size = upload_block_size
        self.write_idx = 0
        self.token_size = token_size
        self.token_format = "I" if self.token_size == 4 else "H"
        self.doc_ends = []
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.save_final_metadata = save_final_metadata

        self.tokens_file = self.output_folder.open(self.filename, mode="wb", block_size=upload_block_size)
        self.loss_file: DataFolderLike | None = None
        if self.save_loss_metadata:
            self.loss_file = self.output_folder.open(f"{self.filename}.loss", mode="wb", block_size=upload_block_size)

    def __len__(self):
        return self.doc_ends[-1] if self.doc_ends else 0

    def close(self):
        """Close the files and save the index."""
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

        if self.save_final_metadata:
            self.write_final_metadata()

    def cleanup(self):
        """Remove the files and the index."""
        self.doc_ends = []
        self.output_folder.rm_file(self.filename)
        if self.loss_file:
            self.output_folder.rm_file(f"{self.filename}.loss")
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

    def write_loss_bytes(self, l_bytes: bytes):
        """Write loss mask to the loss file.

        Args:
          l_bytes: bytes:

        Returns:

        """
        if self.save_loss_metadata:
            self.loss_file.write(l_bytes)

    def write(self, tokens: list[int], loss_values: np.ndarray | None):
        """Write tokens and loss values to the files.

        Args:
            tokens (list[int]): the tokens to write
            loss_values (np.ndarray | None): optional loss values to write
        """
        # get the bytes
        self.write_bytes(struct.pack(f"<%s{self.token_format}" % len(tokens), *tokens))
        if loss_values is not None:
            self.write_loss_bytes(struct.pack("<%s?" % len(loss_values), *loss_values))

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
            sub_rank = 0
            destination = get_output_filename(save_filename, rank, "shuffled", sub_rank)

            new_file = TokenizedFile(
                self.output_folder if not new_output_folder else new_output_folder,
                destination,
                save_loss_metadata=self.save_loss_metadata,
                upload_block_size=self.upload_block_size,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                save_final_metadata=self.save_final_metadata,
                token_size=self.token_size,
            )
            logger.info(f"Shuffling in {destination}...")
            # shuffle doc_id
            total_tokens_written = 0
            for doc_id in ordering:
                # get start and end from the boundaries
                start, end = self.doc_ends[doc_id - 1] if doc_id > 0 else 0, self.doc_ends[doc_id]
                # copy the bytes. each token is token_size bytes
                tokens_file.seek(start * self.token_size)
                new_file.write_bytes(tokens_file.read((end - start) * self.token_size))
                # copy loss values (1 byte per token)
                if loss_file:
                    loss_file.seek(start)
                    new_file.write_loss_bytes(loss_file.read(end - start))
                total_tokens_written += end - start
                if max_tokens_per_file and total_tokens_written > max_tokens_per_file:
                    new_file.close()
                    sub_rank += 1
                    destination = get_output_filename(save_filename, rank, "shuffled", sub_rank)
                    new_file = TokenizedFile(
                        self.output_folder if not new_output_folder else new_output_folder,
                        destination,
                        save_loss_metadata=self.save_loss_metadata,
                        upload_block_size=self.upload_block_size,
                        tokenizer_name_or_path=self.tokenizer_name_or_path,
                        save_final_metadata=self.save_final_metadata,
                        token_size=self.token_size,
                    )
                    logger.info(f"Shuffling in {destination}...")
                    total_tokens_written = 0
            if loss_file:
                loss_file.close()
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


def get_output_filename(save_filename, rank: int, name: str, sub_rank: int = None):
    """Get an output filename for the rank and a sub-step name (unshuffled/shuffled)."""
    if sub_rank is not None:
        return "_".join([x for x in [save_filename, f"{rank:05d}", f"{sub_rank:05d}", f"{name}.ds"] if x])
    return "_".join([x for x in [save_filename, f"{rank:05d}", f"{name}.ds"] if x])


class DocumentTokenizer(PipelineStepWithTokenizer):
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
        save_loss_metadata (bool): save the loss information (default: False)
        shuffle (bool): whether to shuffle documents in the dataset (default: True)
        batch_size (int): batch size for tokenization (default: 1000)
        seed (int): the seed to use for shuffling
        save_final_metadata (bool): whether to save the final metadata (default: True)
        upload_block_size (int | None): the fsspec size of the upload block for remote filesystems (S3)
            You can set this if your s3 uploads are failing because of "Part number must be an integer between 1 and 10000, inclusive".
            Example: 20 * 2**20 (20MB)
    """

    name = "✍️ Writer"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        output_folder: DataFolderLike,
        local_working_dir: DataFolderLike | None = None,
        save_filename: str = None,  # if defined, the final output filename will be this
        tokenizer_name_or_path: str = "gpt2",  # tokenizer to use, from HF or a local
        eos_token: str = "<|endoftext|>",  # whether to add the EOS token after each document
        save_loss_metadata: bool = False,  # save the loss information
        shuffle: bool = True,  # whether to shuffle documents in the dataset,
        batch_size: int = 10000,  # batch size for tokenization
        max_tokens_per_file: int = None,  # max tokens per file to get more (smaller) shuffled output files
        seed: int = None,
        save_final_metadata: bool = True,
        upload_block_size: int | None = None,
        # you can set this if your s3 uploads are failing because of "Part
        # number must be an integer between 1 and 10000, inclusive". Example: 20 * 2**20 (20MB)
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
        self.eos_token = eos_token
        self.save_loss_metadata = save_loss_metadata
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.rand = default_rng(seed)
        self.save_final_metadata = save_final_metadata
        self.upload_block_size = upload_block_size
        self.max_tokens_per_file = max_tokens_per_file

    def get_loss_values(self, document: Document, encoded: "Encoding"):
        """Get the loss mask for the document, if needed.
            A loss mask is defined as a list of tuple of start, end character positions in the string for which we want to ignore the loss.
            This is useful for example when we have a document with a prompt and we want to ignore the loss for the prompt.

        Args:
            document (Document): the document to process
                document metadata can contain a "no_loss_ranges" list of tuple of start, end character positions List[Tuple[int, int]]
            encoded (Encoding): the encoded document
        """
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
        """Tokenize documents with the tokenizer in batches and write the unshuffled tokenized documents to a file.
            We also compute loss values if needed and save them.

        Args:
            data (DocumentsPipeline): the documents to process
            filename (str): the filename to use for the output file
        """
        from tokenizers import Encoding

        unshuff = TokenizedFile(
            self.output_folder if not self.shuffle or not self.local_working_dir else self.local_working_dir,
            filename,
            save_index=not self.shuffle,
            save_loss_metadata=self.save_loss_metadata,
            upload_block_size=self.upload_block_size,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            save_final_metadata=self.save_final_metadata,
            token_size=self.token_size,
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
        unshuf_filename = get_output_filename(self.save_filename, rank, "unshuffled")
        logger.info(f'Tokenizing in "{unshuf_filename}"...')
        outputfile: TokenizedFile = self.write_unshuffled(data, unshuf_filename)
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
