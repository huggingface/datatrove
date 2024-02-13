import math
from dataclasses import dataclass
from functools import partial
from typing import BinaryIO, Generator, List, Union

import numpy as np
from numpy.random import default_rng

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.tokenizer import TokenizedFile


@dataclass
class Chunk:
    """A chunk of a file containing tokenized documents."""

    file_id: int
    start_e: int
    doc_ends: np.ndarray


class FileTokenizerMerger(PipelineStep):
    """Merge/shuffle a folder of tokenized files into a sequence of files with a maximum number of tokens per file.
        This pipeline step is used after the DocumentTokenizer step to merge the tokenized files into a sequence of files.

        NOTE: This pipeline step don't do a full random shuffling accross documents but shuffle chunk which are made of:
            - tokenized files if enough files are provided
            - chunk of tokenized files if not enough files are provided (less than min_chunks_to_shuffle)
        If you want to shuffle the documents in a more extensive way, you can use the DocumentTokenizerMerger
        pipeline step instead but note that DocumentTokenizerMerger does full random shuffling accross documents.

        If you have enough files to process in comparison to the size of your dataset, using this pipeline step
        will be much faster than using DocumentTokenizerMerger.

    Args:
        input_folder (DataFolderLike): the input folder containing the tokenized documents
        output_folder (DataFolderLike): the output folder where to save the merged tokenized documents
        save_filename (str): the filename to use for the merged tokenized documents
        min_chunks_to_shuffle (int): the minimum number of documents per chunk (if we have less input files than this,
            we will split input file in chunk to have at least this number of chunk to shuffle). Default: 10000
        max_tokens_per_file (int): the maximum number of tokens per file. Default: 100GT
        max_tokens (int): the maximum number of tokens to process. Default: -1
        shuffle (bool): whether to shuffle the documents in the dataset. Default: True
        upload_block_size (int): the upload block size to use when saving the tokenized files (used in fsspec with remote filesystems).
            Default: 20MB
        seed (int): the seed to use for the random number generator. Default: None
        save_loss_metadata (bool): whether to save the loss metadata. Default: False
        save_final_metadata (bool): whether to save the final metadata. Default: True
    """

    name = "🗃 Merger"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        save_filename: str,  # if defined, the final output filename will be this
        min_chunks_to_shuffle: int = 10000,  # min number of documents per chunk (if we have less input files than this, we will split input file in chunk to have at least)
        max_tokens_per_file: int = 100e9,  # max number of tokens per file. default: 100GT
        max_tokens: int = -1,  # max number of tokens to process
        shuffle: bool = True,  # whether to shuffle documents in the dataset
        upload_block_size: int = 20 * 2**20,  # 20MB
        # upload_block_size * 10000 must be bigger than 2*max_tokens_per_file,
        # or s3 uploads will fail
        seed: int = None,
        save_loss_metadata: bool = False,
        save_final_metadata: bool = True,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.save_filename = save_filename
        self.max_tokens_per_file = max_tokens_per_file
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.save_loss_metadata = save_loss_metadata
        self.rand = default_rng(seed)
        self.save_final_metadata = save_final_metadata
        self.upload_block_size = upload_block_size
        self.min_chunks_to_shuffle = min_chunks_to_shuffle

    def get_ordering(self, file_chunks: list[Chunk]) -> Union[np.ndarray, List[int]]:
        """Get the ordering of the files to process.
        If shuffle is True, the files are shuffled, otherwise they are returned as is.
        """
        if not self.shuffle:
            return file_chunks
        else:
            perm = self.rand.permutation(range(len(file_chunks)))
            return [file_chunks[i] for i in perm]

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Main method to run the merging of files.
            The world_size must be 1 for this pipeline step merging the results of the previous parallel step.



        Args:
            data: not used we're using as source the input_folder
            rank: int: not used - the rank of the process - we only have one process
            world_size: int: not used — The total number of processes - we only have one process
        """
        assert world_size == 1, "world_size must be 1 for DocumentTokenizerMerger"
        datafiles = self.input_folder.list_files(glob_pattern="*.ds")
        datafiles_index = self.input_folder.list_files(glob_pattern="*.ds.index")
        datafiles_loss = (
            self.input_folder.list_files(glob_pattern="*.ds.loss")
            if self.save_loss_metadata
            else ([None] * len(datafiles))
        )
        assert len(datafiles) == len(datafiles_index) == len(datafiles_loss), (
            f"Mismatch between number of .ds, "
            ".ds.index and/or .ds.loss files"
            f"({len(datafiles)} vs {len(datafiles_index)} vs {len(datafiles_loss)})"
        )

        doc_ends = [load_doc_ends(self.input_folder.open(file, "rb")) for file in datafiles_index]

        # If we have less input files than the min_chunks_to_shuffle, we will split input file in chunk to have at
        # least min_chunks_to_shuffle documents per chunk
        if self.min_chunks_to_shuffle > 0 and len(datafiles) < self.min_chunks_to_shuffle:
            file_splits = math.ceil(self.min_chunks_to_shuffle / len(datafiles))
            # We take care of the edge case where we have less document in a file than the file_splits
            datafiles_chunks = []
            for i in range(len(datafiles)):
                chunk_in_file = min(
                    len(doc_ends[i]), file_splits
                )  # Let's not split in more than the number of documents
                chunk_ends_list = np.array_split(doc_ends[i], chunk_in_file)
                start_e = 0
                for chunk_ends in chunk_ends_list:
                    datafiles_chunks.append(Chunk(file_id=i, start_e=start_e, doc_ends=chunk_ends))
                    start_e = chunk_ends[-1]
        else:
            file_splits = 1
            datafiles_chunks = [Chunk(file_id=i, start_e=0, doc_ends=doc_ends[i]) for i in range(len(datafiles))]

        # Now we mix the order of the chunks (full files if enough files, or chunks of files if we have less than min_chunks_to_shuffle)
        shuffled_datafiles_chunks = self.get_ordering(datafiles_chunks)

        # Now we gather then again in the new shuffled order
        # in files of about max_tokens_per_file tokens
        chunks_per_files: List[
            List[Chunk]
        ] = []  # List of length number of output_files (so that max tokens per file is respected)
        chunks_per_file = []  # For each output file, list the chunks of input files that will be merged in this output file
        current_tokens_in_file = 0
        total_tokens = 0
        for file_chunk in shuffled_datafiles_chunks:
            file_id = file_chunk.file_id
            doc_ends = file_chunk.doc_ends
            start_e = file_chunk.start_e
            if doc_ends[-1] - start_e + current_tokens_in_file < self.max_tokens_per_file:
                # The full chunk fits in the current file given our max_tokens_per_file budget
                chunks_per_file.append(Chunk(file_id=file_id, doc_ends=doc_ends, start_e=start_e))
                current_tokens_in_file += doc_ends[-1] - start_e
            else:
                # We have at least one new file boundary in this chunk, so we'll we need to split it
                while 0 < self.max_tokens_per_file <= doc_ends[-1] - start_e + current_tokens_in_file:
                    switching_index = np.argmax(doc_ends - start_e + current_tokens_in_file > self.max_tokens_per_file)

                    # We include the document that make us bigger than the max_tokens_per_file budget as well (easier to manage)
                    chunks_per_file.append(
                        Chunk(file_id=file_id, doc_ends=doc_ends[: switching_index + 1], start_e=start_e)
                    )
                    current_tokens_in_file += doc_ends[switching_index] - start_e
                    chunks_per_files.append(chunks_per_file)
                    total_tokens += current_tokens_in_file

                    # Start a new file with the remaining part of the chunk
                    current_tokens_in_file = 0
                    chunks_per_file = []
                    start_e = doc_ends[switching_index]
                    doc_ends = doc_ends[
                        switching_index + 1 :
                    ]  # We remove the first part of the document from the list and test again
                    if len(doc_ends) == 0:
                        # We have no more document in this chunk to add to this file
                        break
                if len(doc_ends) > 0:
                    # We still have one of this document left to add to the current new file
                    chunks_per_file.append(Chunk(file_id=file_id, doc_ends=doc_ends, start_e=start_e))
                    current_tokens_in_file += doc_ends[-1] - start_e

            # We stop anyway if we have reached the max_tokens budget
            if 0 < self.max_tokens <= total_tokens:
                break
        if chunks_per_file:
            chunks_per_files.append(chunks_per_file)
            total_tokens += current_tokens_in_file

        tokenizer_name = None
        if self.save_final_metadata:
            if self.input_folder.isfile(f"{datafiles[0]}.metadata"):
                with self.input_folder.open(f"{datafiles[0]}.metadata", "rt") as f:
                    tokenizer_name = f.read().splitlines()[0]

        file_ct = 0
        for chunks_per_file in chunks_per_files:
            output_file = TokenizedFile(
                output_folder=self.output_folder,
                filename=f"{file_ct:03d}_{self.save_filename}.ds",
                save_loss_metadata=self.save_loss_metadata,
                upload_block_size=self.upload_block_size,
            )
            # Get all the data readers for the files we need to merge in this merge file
            chunks_token_inputs = [
                get_data_reader(
                    self.input_folder.open(datafiles[d.file_id]), doc_ends=d.doc_ends, nb_bytes=2, start_e=d.start_e
                )
                for d in chunks_per_file
            ]
            chunks_loss_inputs = (
                [
                    get_data_reader(
                        self.input_folder.open(datafiles_loss[d.file_id]),
                        doc_ends=d.doc_ends,
                        nb_bytes=1,
                        start_e=d.start_e,
                    )
                    for d in chunks_per_file
                ]
                if self.save_loss_metadata
                else None
            )

            for chunk_token in chunks_token_inputs:
                for tokens in chunk_token:
                    output_file.write_bytes(tokens)
                    self.stat_update("tokens", value=len(tokens) // 2)
            if self.save_loss_metadata:
                for chunk_loss in chunks_loss_inputs:
                    for loss in chunk_loss:
                        output_file.write_loss_bytes(loss)

            output_file.close()
            file_ct += 1
            if self.save_final_metadata:
                output_file.save_final_metadata(tokenizer_name)


class DocumentTokenizerMerger(PipelineStep):
    """Merge/shuffle a folder of tokenized files into a sequence of files with a maximum number of tokens per file.
        This pipeline step is used after the DocumentTokenizer step to merge the tokenized files into a sequence of files.

        WARNING: This pipeline step involves accessing multiple files in random order on the filesystem, which can be
        slow on some filesystems (e.g. S3). It is recommended to use a local filesystem for the input and output folders.

        Documents are typically shuffled inside each separate files during the first step. In this second step, we shuffle
        again the order of the documents.

    Args:
        input_folder (DataFolderLike): the input folder containing the tokenized documents
        output_folder (DataFolderLike): the output folder where to save the merged tokenized documents
        save_filename (str): the filename to use for the merged tokenized documents
        max_tokens_per_file (int): the maximum number of tokens per file. Default: 100GT
        max_tokens (int): the maximum number of tokens to process. Default: -1
        shuffle (bool): whether to shuffle the documents in the dataset. Default: True
        upload_block_size (int): the upload block size to use when saving the tokenized files (used in fsspec with remote filesystems).
            Default: 20MB
        seed (int): the seed to use for the random number generator. Default: None
        save_loss_metadata (bool): whether to save the loss metadata. Default: False
        save_final_metadata (bool): whether to save the final metadata. Default: True
    """

    name = "🗃 Merger"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        save_filename: str,  # if defined, the final output filename will be this
        max_tokens_per_file: int = 100e9,  # max number of tokens per file. default: 100GT
        max_tokens: int = -1,  # max number of tokens to process
        shuffle: bool = True,  # whether to shuffle documents in the dataset
        upload_block_size: int = 20 * 2**20,  # 20MB
        # upload_block_size * 10000 must be bigger than 2*max_tokens_per_file,
        # or s3 uploads will fail
        seed: int = None,
        save_loss_metadata: bool = False,
        save_final_metadata: bool = True,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.save_filename = save_filename
        self.max_tokens_per_file = max_tokens_per_file
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.save_loss_metadata = save_loss_metadata
        self.rand = default_rng(seed)
        self.save_final_metadata = save_final_metadata
        self.upload_block_size = upload_block_size

    def get_ordering(self, all_doc_ends):
        """

        Args:
          all_doc_ends:

        Returns:

        """
        doc_ids = np.concatenate([np.ones(len(doc_ends), dtype=int) * i for i, doc_ends in enumerate(all_doc_ends)])
        return doc_ids if not self.shuffle else self.rand.permutation(doc_ids)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Main method to run the merging of files.
            The world_size must be 1 for this pipeline step merging the results of the previous parallel step.

        Args:
            data: DocumentsPipeline
                The data to be processed as a Generator typically created by a Reader initial pipeline step
            rank: int
                The rank of the process
            world_size: int
                The total number of processes
        """
        assert world_size == 1, "world_size must be 1 for DocumentTokenizerMerger"
        datafiles = self.input_folder.list_files(glob_pattern="*.ds")
        datafiles_index = self.input_folder.list_files(glob_pattern="*.ds.index")
        datafiles_loss = (
            self.input_folder.list_files(glob_pattern="*.ds.loss")
            if self.save_loss_metadata
            else ([None] * len(datafiles))
        )
        assert len(datafiles) == len(datafiles_index) == len(datafiles_loss), (
            f"Mismatch between number of .ds, "
            ".ds.index and/or .ds.loss files"
            f"({len(datafiles)} vs {len(datafiles_index)} vs {len(datafiles_loss)})"
        )

        doc_ends = [load_doc_ends(self.input_folder.open(file, "rb")) for file in datafiles_index]
        token_inputs = list(
            map(partial(get_data_reader, nb_bytes=2), self.input_folder.open_files(datafiles), doc_ends)
        )
        loss_inputs = (
            list(map(partial(get_data_reader, nb_bytes=1), self.input_folder.open_files(datafiles_loss), doc_ends))
            if self.save_loss_metadata
            else None
        )

        tokenizer_name = None
        if self.save_final_metadata:
            if self.input_folder.isfile(f"{datafiles[0]}.metadata"):
                with self.input_folder.open(f"{datafiles[0]}.metadata", "rt") as f:
                    tokenizer_name = f.read().splitlines()[0]

        ordering = self.get_ordering(doc_ends)

        file_ct = 0
        output_file = TokenizedFile(
            output_folder=self.output_folder,
            filename=f"{file_ct:03d}_{self.save_filename}.ds",
            save_loss_metadata=self.save_loss_metadata,
            upload_block_size=self.upload_block_size,
        )
        for input_file_id in ordering:
            if 0 < self.max_tokens <= self.stats["tokens"].total:
                break
            if 0 < self.max_tokens_per_file <= len(output_file):
                output_file.close()
                file_ct += 1
                if self.save_final_metadata:
                    output_file.save_final_metadata(tokenizer_name)
                output_file = TokenizedFile(
                    output_folder=self.output_folder,
                    filename=f"{file_ct:03d}_{self.save_filename}.ds",
                    save_loss_metadata=self.save_loss_metadata,
                    upload_block_size=self.upload_block_size,
                )
            # copy tokens and loss
            tokens = next(token_inputs[input_file_id])
            output_file.write_bytes(tokens)
            if loss_inputs:
                output_file.write_loss_bytes(next(loss_inputs[input_file_id]))
            self.stat_update("tokens", value=len(tokens) // 2)
        # cleanup
        output_file.close()
        if self.save_final_metadata:
            output_file.save_final_metadata(tokenizer_name)
            # save final total metadata file
            output_file.save_final_metadata(
                tokenizer_name, self.stats["tokens"].total, filename=f"{self.save_filename}.ds"
            )
        output_file.close()


def load_doc_ends(file: BinaryIO) -> np.ndarray:
    """Load the document ends from a file.

    Args:
        file: BinaryIO
            The file to read from

    Returns:
        np.ndarray
            The document ends: 1-D array of uint64 of length equal to the number of documents
                Each element is the index of the end of a document in the file (in tokens)
    """
    with file as f:
        return np.frombuffer(f.read(), dtype=np.uint64)


def get_data_reader(file: BinaryIO, doc_ends: list, nb_bytes: int, start_e: int = 0) -> Generator[bytes, None, None]:
    """Get a reader for the data in the file given a list of document ends and a number of bytes per element.
        The reader will yield the data for each document in the file.

    Args:
        file: BinaryIO
            The file to read from
        doc_ends: list
            The list of document ends in the file
        nb_bytes: int
            The number of bytes per token
        start_e: int
            The starting index (optional - default: 0)
    """
    with file as f:
        if start_e != 0:
            f.seek(int(start_e) * nb_bytes)
        for r_e in doc_ends:
            yield f.read((int(r_e) - int(start_e)) * nb_bytes)
            start_e = r_e
