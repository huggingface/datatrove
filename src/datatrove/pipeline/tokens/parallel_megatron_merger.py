import pickle
import itertools
import multiprocessing
from functools import partial
from typing import BinaryIO, Generator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, Process, Manager

import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
from loguru import logger

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.megatron_tokenizer import TokenizedFile, load_doc_ends_indices
from datatrove.utils.threaded_pbar import ThreadedProgressBar


class ParaMegatronTokenizerMerger(PipelineStep):
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
        num_workers: int=1,
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
        self.num_workers = num_workers

    def get_ordering(self, all_doc_ends):
        """

        Args:
          all_doc_ends:

        Returns:

        """
        doc_ids = np.concatenate([np.ones(len(doc_ends), dtype=int) * i for i, doc_ends in enumerate(all_doc_ends)])
        return doc_ids if not self.shuffle else self.rand.permutation(doc_ids)

    @staticmethod
    def write_tokenized_file(
            bucket_idx,
            queue,
            extra_args,
            ):
        logger.info(f"Running {bucket_idx=}.")
        extra_args = pickle.loads(extra_args)
        data_buckets            =  extra_args["data_buckets"]
        output_folder           =  extra_args["output_folder"]
        save_filename           =  extra_args["save_filename"]
        suffix                  =  extra_args["suffix"]
        upload_block_size       =  extra_args["upload_block_size"]
        tokenizer_name_or_path  =  extra_args["tokenizer_name_or_path"]
        save_final_metadata     =  extra_args["save_final_metadata"]
        token_size              =  extra_args["token_size"]
        datafiles               =  extra_args["datafiles"]
        doc_indices             =  extra_args["doc_indices"]
        doc_lengths             =  extra_args["doc_lengths"]
        doc_ends                =  extra_args["doc_ends"]

        data_bucket = data_buckets[bucket_idx]
        output_file = TokenizedFile(
            output_folder=output_folder,
            filename=f"{bucket_idx:03d}_{save_filename}{suffix}",
            upload_block_size=upload_block_size,
            tokenizer_name_or_path=tokenizer_name_or_path,
            save_final_metadata=save_final_metadata,
            token_size=token_size,
        )
        tokens_processed = 0
        for ds_idx, doc_idx in data_bucket:
            # copy tokens
            doc_end = doc_ends[ds_idx][doc_idx]
            doc_start = doc_end - doc_lengths[ds_idx][doc_idx]
            tokens = read_data_slice(
                    file=datafiles[ds_idx],
                    doc_start=doc_start,
                    doc_end=doc_end,
                    nb_bytes=token_size,
                    )
            output_file.write_bytes(tokens)
            output_file.doc_indices.append(doc_indices[ds_idx][doc_idx]) 
            tokens_processed += len(tokens) // token_size
            queue.put(1)
        # cleanup
        output_file.close()
        return tokens_processed

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
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            assert multiprocessing.get_start_method() == "spawn", "Multiprocessing start method must be spawn"

        assert world_size == 1, "world_size must be 1 for DocumentTokenizerMerger"
        datafiles = self.input_folder.list_files(glob_pattern=f"*{self.suffix}")
        datafiles_index = self.input_folder.list_files(glob_pattern="*.idx")
        assert len(datafiles) == len(datafiles_index), (
            f"Mismatch between number of {self.suffix}, "
            "and .idxfiles"
            f"({len(datafiles)} vs {len(datafiles_index)}"
        )
        # datafiles = self.input_folder.get_shard(rank, world_size, recursive=True, glob_pattern=f"*{self.suffix}")
        logger.info(f"Use first 10 files for testing purpose.")
        datafiles = datafiles[:10]

        tokenizer_name_or_path, token_size = None, 2
        if self.save_final_metadata:
            if self.input_folder.isfile(f"{datafiles[0]}.metadata"):
                with self.input_folder.open(f"{datafiles[0]}.metadata", "rt") as f:
                    tokenizer_name_or_path = f.read().splitlines()[0]
                    if "|" in tokenizer_name_or_path:
                        tokenizer_name_or_path, token_size = tokenizer_name_or_path.split("|")
                        token_size = int(token_size)

        logger.info(f"Loading token indices file.")
        doc_lengths, doc_ends, doc_indices, data_indicies = [], [], [], []
        for i, file in tqdm(enumerate(datafiles),
                desc="Data files",
                unit="files",
                total=len(datafiles),
                disable=not self.progress,
                ):
          doc_end, doc_index = load_doc_ends_indices(self.input_folder._join(file.replace(self.suffix, ".idx")))
          doc_len = doc_end.copy()
          doc_len[1:] -= doc_end[:-1]
          doc_lengths.append(doc_len)
          doc_ends.append(doc_end)
          doc_indices.append(doc_index.tolist())
          data_indicies.extend(zip(itertools.repeat(i),range(len(doc_end))))

        # plan shuffle
        logger.info(f"Planning document shuffle and bucket split.")
        data_buckets = []
        bucket_size = 0
        data_bucket = []
        for ds_idx, doc_idx in tqdm(self.rand.permutation(data_indicies, axis=0),
                desc="Planning documents",
                unit="documents",
                total=len(data_indicies),
                disable=not self.progress,
                ):
            if bucket_size + doc_lengths[ds_idx][doc_idx] > self.max_tokens_per_file:
                data_buckets.append(data_bucket)
                bucket_size = 0
                data_bucket = []
            bucket_size += doc_lengths[ds_idx][doc_idx]
            data_bucket.append((ds_idx, doc_idx))
        if data_bucket:
            data_buckets.append(data_bucket)
        n_buckets = len(data_buckets)
        logger.info(f"Split tokens into {n_buckets} dataset files.")

        num_workers = min(n_buckets, self.num_workers)
        logger.info(f"Merge documents in parallel with {num_workers} workers.")
        with Pool(num_workers) as pool:
            queue = (manager := Manager()).Queue()
            with ThreadedProgressBar(
                    desc="Merging documents",
                    unit="documents",
                    total=len(data_indicies),
                    disable=not self.progress,
                    pbar_timeout=5e-5,
                    worker_type=Process,
                    queue=queue,
                    ) as pbar, \
                    tqdm(desc="Datasets", unit="datasets", total=n_buckets, disable=not self.progress) as dspbar:
                extra_args = pickle.dumps(dict(
                        data_buckets=data_buckets,
                        output_folder=self.output_folder,
                        save_filename=self.save_filename,
                        suffix=self.suffix,
                        upload_block_size=self.upload_block_size,
                        tokenizer_name_or_path=tokenizer_name_or_path,
                        save_final_metadata=self.save_final_metadata,
                        token_size=token_size,
                        datafiles=[self.input_folder._join(datafile) for datafile in datafiles],
                        doc_indices=doc_indices,
                        doc_lengths=doc_lengths,
                        doc_ends=doc_ends,
                        ))
                func = partial(self.write_tokenized_file,
                        queue=queue,
                        extra_args=extra_args,
                        )
            #  with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                #  futures = {pool.submit(func(i)): i for i in range(n_buckets)}
                imap_unordered_it = pool.imap_unordered(func, range(n_buckets))
                logger.info(f"Jobs submitted..")
                #  for future in as_completed(futures):
                #      dspbar.update(1)
                #      self.stat_update("tokens", value=future.result())
                for result in imap_unordered_it:
                    dspbar.update(1)
                    self.stat_update("tokens", value=result)

        # cleanup
        logger.info(f"Clean up.")
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


def read_data_slice(
    file: BinaryIO, doc_start:int, doc_end: int, nb_bytes: int = 1
) -> Generator[bytes, None, None]:
    """Get a reader for the data in the file given a list of document ends and a number of bytes per element.
        The reader will yield the data for each document in the file.

    Args:
        file: BinaryIO
            The file to read from
        doc_start: int
            The start index of document in the file
        doc_end: int
            The end index of document in the file
        nb_bytes: int
            The number of bytes per token
    """
    with open(file, "rb") as f:
        f.seek(int(doc_start) * nb_bytes)
        return f.read((doc_end - doc_start) * nb_bytes)
