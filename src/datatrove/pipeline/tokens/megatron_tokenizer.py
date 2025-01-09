import struct
from typing import List

import numpy as np

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.utils.batching import batched
from datatrove.utils.logging import logger
from datatrove.utils.tokenization import PipelineStepWithTokenizer


_INDEX_HEADER = b"MMIDIDX\x00\x00"


class MegatronTokenizedFile:
    """Class to write tokenized documents to local/remote folders.
        Handles writing the tokenized document and an index file with specific metadata.

        Inspired by https://github.com/NVIDIA/NeMo/blob/062532770dbe790e73637dcd0926d964628cbaa5/nemo/collections/nlp/data/language_modeling/megatron/indexed_dataset.py#L380-L591

    Args:
        output_folder (DataFolderLike): the output folder where to save the tokenized documents
        filename (str): the filename to use
        upload_block_size (int): the fsspec size of the upload block for remote filesystems (S3)
        token_size (int): size of each token, in bytes

    """

    def __init__(
        self,
        output_folder: DataFolderLike,
        filename: str,
        upload_block_size: int | None = None,
        token_size: int = 2,
    ):
        self.output_folder = get_datafolder(output_folder)
        self.sequence_lengths = []
        self.filename = filename
        self.upload_block_size = upload_block_size
        self.token_size = token_size
        self.token_dtype = np.int32 if token_size == 4 else np.uint16
        self.token_dtype_code = (
            4 if token_size == 4 else 8
        )  # NOTE(tj.solergibert) Megatron needs this dtype code in the .idx file | https://github.com/NVIDIA/Megatron-LM/blob/64cbae55ac85cd73fbadbc3c0d715c8123c5e13b/megatron/core/datasets/indexed_dataset.py#L41
        self.document_indices = [0]  # NOTE(tj.solergibert) Megatron needs this document_indices field

        self.bin_file = self.output_folder.open(f"{self.filename}.bin", mode="wb", block_size=upload_block_size)

    def __len__(self):
        return sum(self.sequence_lengths) if self.sequence_lengths else 0

    def close(self):
        """Close the files and save the .bin & .idx files"""

        # Save .bin file
        self.bin_file.close()

        # Save .idx file
        # This file has:
        ## 9 Bytes from the _INDEX_HEADER
        ## 8 Byte of metadata (Just a "1")
        ## 1 Byte from the token_dtype_code
        ## 8 Bytes from the number of sequences
        ## 8 Bytes from the number of documents
        ## 8 Bytes from the initial document index
        ## 20 Bytes for every sequence/document
        ### 4 Bytes from the sequence length
        ### 8 bytes from the sequence offset
        ### 8 Bytes from the document index
        # So, if the .bin contains tokens from 35000 text sequences/documents, the .idx will have
        # 9+8+1+8+8+8+20*35000 = 700042 Bytes
        self.idx_file = self.output_folder.open(f"{self.filename}.idx", mode="wb", block_size=self.upload_block_size)
        # Index Header
        self.idx_file.write(_INDEX_HEADER)
        # Version
        self.idx_file.write(struct.pack("<Q", 1))
        # Numeric code for the DType
        self.idx_file.write(struct.pack("<B", self.token_dtype_code))

        sequence_pointers = self._sequence_pointers(self.sequence_lengths)

        # Number of sequences in the dataset
        sequence_count = len(self.sequence_lengths)
        self.idx_file.write(struct.pack("<Q", sequence_count))

        # Number of documents in the dataset
        document_count = len(self.document_indices)
        self.idx_file.write(struct.pack("<Q", document_count))

        # Number of tokens per sequence
        sequence_lengths = np.array(self.sequence_lengths, dtype=np.int32)
        self.idx_file.write(sequence_lengths.tobytes(order="C"))
        del sequence_lengths

        # Byte offsets for all sequences
        sequence_pointers = np.array(sequence_pointers, dtype=np.int64)
        self.idx_file.write(sequence_pointers.tobytes(order="C"))
        del sequence_pointers

        # Sequence indices marking the end of each document
        document_indices = np.array(self.document_indices, dtype=np.int64)
        self.idx_file.write(document_indices.tobytes(order="C"))

        # Close
        self.idx_file.close()

    def _sequence_pointers(self, sequence_lengths: List[int]) -> List[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (List[int]): The length of each sequence

        Returns:
            List[int]: The pointer to the beginning of each sequence
        """
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += length * self.token_size
        return list_ptr

    def write(self, tokens: list[int]):
        """Write tokens to the files.

        Args:
            tokens (list[int]): the tokens to write
        """

        np_array = np.array(tokens, dtype=self.token_dtype)
        self.bin_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.append(np_array.size)
        self.document_indices.append(
            len(self.sequence_lengths)
        )  # NOTE(tj.solergibert) Megatron needs this document_indices field


def get_output_filename(save_filename, rank: int, name: str, sub_rank: int = None):
    """Get an output filename for the rank."""
    return "_".join([x for x in [save_filename, f"{rank:05d}", f"{name}"] if x])


class MegatronDocumentTokenizer(PipelineStepWithTokenizer):
    """Tokenize the documents in the pipeline using the HuggingFace fast tokenizers library.
        This pipeline step saves the tokenized documents locally/remotely in a set of files.

    Args:
        output_folder (DataFolderLike): the output folder where to save the tokenized documents
        save_filename (str): the filename to use for the final tokenized files (default: None – use the default filename)
        tokenizer_name_or_path (str): the name or path of the tokenizer to use, from the HuggingFace tokenizers library (default: "gpt2")
        eos_token (str): whether to add the EOS token after each document (default: "<|endoftext|>")
        batch_size (int): batch size for tokenization (default: 1000)
        upload_block_size (int | None): the fsspec size of the upload block for remote filesystems (S3)
            You can set this if your s3 uploads are failing because of "Part number must be an integer between 1 and 10000, inclusive".
            Example: 20 * 2**20 (20MB)
    """

    name = "✍️ Writer"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        output_folder: DataFolderLike,
        save_filename: str = None,  # If defined, the final output filename will be this
        tokenizer_name_or_path: str = "gpt2",  # Tokenizer to use, from HF or a local
        eos_token: str = "<|endoftext|>",  # Whether to add the EOS token after each document
        batch_size: int = 10000,  # Batch size for tokenization
        upload_block_size: int | None = None,
        # You can set this if your s3 uploads are failing because of "Part
        # number must be an integer between 1 and 10000, inclusive". Example: 20 * 2**20 (20MB)
    ):
        super().__init__(tokenizer_name_or_path, eos_token)

        self.output_folder = get_datafolder(output_folder)
        self.save_filename = save_filename
        self.batch_size = batch_size
        self.upload_block_size = upload_block_size

    def write_tokens(self, data: DocumentsPipeline, filename: str):
        """Tokenize documents with the tokenizer in batches and write the tokens to a file.

        Args:
            data (DocumentsPipeline): the documents to process
            filename (str): the filename to use for the output file
        """
        from tokenizers import Encoding

        unshuff = MegatronTokenizedFile(
            self.output_folder,
            filename,
            upload_block_size=self.upload_block_size,
            token_size=self.token_size,
        )
        # Tokenize document's text in batches to go faster
        for batch in batched(data, self.batch_size):
            with self.track_time(unit="batch"):
                encoded_batch: list[Encoding] = self.tokenizer.encode_batch([document.text for document in batch])
                for encoded in encoded_batch:
                    tokens = encoded.ids
                    # Write bytes to disk
                    unshuff.write(tokens)
                    # Save stats
                    self.stat_update("tokens", value=len(tokens))
        unshuff.close()
        return unshuff

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Main method to run the Megatron tokenization.
            We first batch tokenize the documents and write them to a file.

        Args:
            data: DocumentsPipeline
                The data to be processed as a Generator typically created by a Reader initial pipeline step
            rank: int
                The rank of the process
            world_size: int
                The total number of processes
        """
        prefix_filename = get_output_filename(self.save_filename, rank, "tokens")
        logger.info(f'Tokenizing in "{prefix_filename}"...')
        outputfile: MegatronTokenizedFile = self.write_tokens(data, prefix_filename)
        if len(outputfile) == 0:
            logger.warning("No data saved.")
            return
