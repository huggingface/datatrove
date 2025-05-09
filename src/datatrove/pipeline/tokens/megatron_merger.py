from functools import partial
from typing import BinaryIO, Generator

import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
from loguru import logger

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.megatron_tokenizer import TokenizedFile, load_doc_ends_indices


class MegatronTokenizerMerger(PipelineStep):
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
        save_final_metadata (bool): whether to save the final metadata. Default: True
    """

    name = "🗃 Megatron Merger"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        save_filename: str,  # if defined, the final output filename will be this
        max_tokens_per_file: int = 1024**3,  # max number of tokens per file. default: 1GT
        max_tokens: int = -1,  # max number of tokens to process
        shuffle: bool = True,  # whether to shuffle documents in the dataset
        upload_block_size: int = 20 * 2**20,  # 20MB
        # upload_block_size * 10000 must be bigger than 2*max_tokens_per_file,
        # or s3 uploads will fail
        seed: int = None,
        save_final_metadata: bool = True,
        progress: bool = True,
        suffix: str = ".npy",
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.save_filename = save_filename
        self.max_tokens_per_file = max_tokens_per_file
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.rand = default_rng(seed)
        self.save_final_metadata = save_final_metadata
        self.upload_block_size = upload_block_size
        self.progress = progress
        self.suffix = suffix

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
        datafiles = self.input_folder.list_files(glob_pattern=f"*{self.suffix}")
        datafiles_index = self.input_folder.list_files(glob_pattern="*.idx")
        assert len(datafiles) == len(datafiles_index), (
            f"Mismatch between number of {self.suffix}, "
            "and .idxfiles"
            f"({len(datafiles)} vs {len(datafiles_index)}"
        )

        tokenizer_name_or_path, token_size = None, 2
        if self.save_final_metadata:
            if self.input_folder.isfile(f"{datafiles[0]}.metadata"):
                with self.input_folder.open(f"{datafiles[0]}.metadata", "rt") as f:
                    tokenizer_name_or_path = f.read().splitlines()[0]
                    if "|" in tokenizer_name_or_path:
                        tokenizer_name_or_path, token_size = tokenizer_name_or_path.split("|")
                        token_size = int(token_size)

        doc_ends, doc_indices = [], []
        for file in datafiles:
          doc_end, doc_index = load_doc_ends_indices(self.input_folder._join(file.replace(self.suffix, ".idx")))
          doc_ends.append(doc_end)
          doc_indices.append(doc_index.tolist())

        token_inputs = list(
            map(partial(get_data_reader, nb_bytes=token_size), self.input_folder.open_files(datafiles), doc_ends)
        )

        # Token source file ordering
        ordering = self.get_ordering(doc_ends)

        file_ct = 0
        output_file = TokenizedFile(
            output_folder=self.output_folder,
            filename=f"{file_ct:03d}_{self.save_filename}{self.suffix}",
            upload_block_size=self.upload_block_size,
            tokenizer_name_or_path=tokenizer_name_or_path,
            save_final_metadata=self.save_final_metadata,
            token_size=token_size,
        )
        for input_file_id in tqdm(
            ordering, desc="Merging documents", unit="documents", total=len(ordering), disable=not self.progress
        ):
            if 0 < self.max_tokens <= self.stats["tokens"].total:
                break
            if 0 < self.max_tokens_per_file <= len(output_file):
                output_file.close()
                file_ct += 1
                output_file = TokenizedFile(
                    output_folder=self.output_folder,
                    filename=f"{file_ct:03d}_{self.save_filename}{self.suffix}",
                    upload_block_size=self.upload_block_size,
                    tokenizer_name_or_path=tokenizer_name_or_path,
                    save_final_metadata=self.save_final_metadata,
                    token_size=token_size,
                )
            # copy tokens
            tokens = next(token_inputs[input_file_id])
            output_file.write_bytes(tokens)
            output_file.doc_indices.append(doc_indices[input_file_id].pop(0)) 
            self.stat_update("tokens", value=len(tokens) // token_size)
        # cleanup
        output_file.close()
        if self.save_final_metadata:
            # save final total metadata file
            output_file.write_final_metadata(self.stats["tokens"].total, filename=f"{self.save_filename}.metadata")


def get_data_reader(
    file: BinaryIO, doc_ends: list, nb_bytes: int = 1, start_e: int = 0
) -> Generator[bytes, None, None]:
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
            yield f.read((r_e - start_e) * nb_bytes)
            start_e = r_e
