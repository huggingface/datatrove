"""
URL based deduplication.
"""

import contextlib
import heapq
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import BinaryIO, Callable, Generator, Iterator

import numpy as np
from fsspec.spec import AbstractBufferedFile
from tqdm import tqdm

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.binaryio import read_np_from_file, read_tuples_from_file
from datatrove.utils.hashing import HashConfig, create_hash_func_from_bytes
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import ExtensionHelperSD, StatHints

from ..writers.disk_base import DiskWriter


@dataclass
class KeepRecordsKOccurrancesConfig:
    """
    Args:
        url_normalizer: Callable[[str], str] Normalize the url, e.g. remove query parameters
    """

    bytes_getter: Callable[[Document], bytes]
    k: int
    hash_config: HashConfig = field(default_factory=HashConfig)


@dataclass(order=False)
class HashSig:
    hash_value: int
    doc_id: int
    file_id: int
    file_stem: str

    def __lt__(self, other: "HashSig") -> bool:
        return (self.hash_value, self.doc_id) < (
            other.hash_value,
            other.doc_id,
        )


def get_sig_dtype(config: HashConfig) -> np.dtype:
    return np.dtype([("hash", config.np_dtype), ("doc", "<u4")])


class KeepRecordsKOccurrancesSignature(PipelineStep):
    """KeepRecordsKOccurrancesSignature: First pipeline step
        Creates a signature for url in each document. Each HashSig has n hashes, the doc id. Before saving
        them the hashes are sorted based on (hash, doc_id).

    Args:
        output_folder: folder where signatures are saved
        finder_workers: number of workers used in finder stage of deduplication
        config: configuration for the dedup
    """

    type = "🫂 - DEDUP"
    name = "💥 keep records k occurrances signature"

    def __init__(
        self,
        output_folder: DataFolderLike,
        config: KeepRecordsKOccurrancesConfig,
        finder_workers: int = 1,
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        if finder_workers <= 0:
            raise ValueError("finder_workers must be >= 1")
        elif finder_workers > 1:
            logger.warning(f"Remember to also set the number of tasks of the finder block to {finder_workers=}!")
        self.finder_workers = finder_workers
        self.config = config
        self.hash_fc = create_hash_func_from_bytes(self.config.hash_config)

    def save_hashes(self, rank: int, signatures):
        sig_dtype = get_sig_dtype(self.config.hash_config)
        signatures = np.array(signatures, dtype=sig_dtype)

        signatures.sort(axis=0)

        # Same code as in sentence_dedup
        hashes_per_worker = self.config.hash_config.max // self.finder_workers
        left_idx = 0
        for hash_i in range(self.finder_workers):
            with self.output_folder.open(
                f"{hash_i:04d}/{rank:05d}{ExtensionHelperSD.stage_1_signature}",
                mode="wb",
            ) as f:
                # last bucket needs to have everything
                right_hash = (
                    (hash_i + 1) * hashes_per_worker if hash_i != self.finder_workers - 1 else np.iinfo(np.uint64).max
                )
                # find last hash that goes in this bucket. This obeys the following rule:
                # signatures['hash'][right_idx - 1] <= right_hash <= signatures['hash'][right_idx]
                right_idx = left_idx + signatures["hash"][left_idx:].searchsorted(right_hash, side="right")
                # save to file
                if right_idx > left_idx:
                    bts = signatures[left_idx:right_idx].tobytes()
                    f.write(bts)
                left_idx = right_idx
                # we've reached the end of our data
                if right_idx >= len(signatures):
                    break

    def get_hashes(self, doc: Document, doc_idx: int) -> list[None] | list[tuple[int, int, int]]:
        bytes_to_hash: bytes = (
            self.config.bytes_getter(doc)
        )
        hashes = [(self.hash_fc(bytes_to_hash), doc_idx)]

        return hashes

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        signatures = []
        for doc_idx, doc in enumerate(data):
            with self.stats.time_stats:
                self.stat_update(StatHints.total)
                signatures.extend(self.get_hashes(doc, doc_idx))
        self.save_hashes(rank, signatures)


def read_sigs(
    file: AbstractBufferedFile,
    file_id: int,
    hash_config: HashConfig,
    lines_to_buffer: int = 5,
) -> Generator[HashSig, None, None]:
    last = None
    line_format = f"{hash_config.struct_format}I"
    with file as f:
        file_stem = Path(f.path).name.removesuffix(ExtensionHelperSD.stage_1_signature)
        for data in read_tuples_from_file(f, line_format, lines_to_buffer=lines_to_buffer):
            assert last is None or data[0] >= last, f"Hash order error. {f.tell()=}, {data[0]=}, {last=}"
            last = data[0]
            yield HashSig(
                file_id=file_id,
                file_stem=file_stem,
                hash_value=data[0],
                doc_id=data[1],
                )


class KeepRecordsKOccurrancesFindKOccurrances(PipelineStep):
    """KeepRecordsKOccurrancesFindKOccurrances: Second pipeline step
        Reads signatures from each input data_folder into separate queues.
        It iteratively finds the minimum hash value present at the top of any queue.
        If exactly k_occurrences folders have this minimum hash at their top,
        the corresponding records (doc_ids) are written to dedicated output folders,
        one for each input data folder.
        Otherwise, the records with the minimum hash are discarded (popped from their queues).

    Args:
        data_folders: A list of data folders where stage 1 signatures are saved.
        output_folders: A list of output folders, corresponding 1:1 with data_folders,
                        where the doc_ids of records meeting the k_occurrences criteria are saved.
        config: Configuration for the dedup (mainly hash settings).
        lines_to_buffer: Number of lines to buffer when reading signature files.
    """

    type = "🫂 - DEDUPS"
    name = "💥 keep records k occurrences stage 2"

    def __init__(
        self,
        data_folders: list[DataFolderLike],
        output_folders: list[DataFolderLike],
        config: KeepRecordsKOccurrancesConfig,
        lines_to_buffer: int = 5,
    ):
        super().__init__()
        if not data_folders:
             raise ValueError("data_folders list cannot be empty.")
        if len(data_folders) != len(output_folders):
             raise ValueError(f"Number of data_folders ({len(data_folders)}) must match "
                              f"number of output_folders ({len(output_folders)}).")
        if config.k < 1 or config.k > len(data_folders):
             raise ValueError(f"k must be between 1 and {len(data_folders)}, got {config.k}")
        self.data_folders = [get_datafolder(folder) for folder in data_folders]
        self.output_folders = [get_datafolder(folder) for folder in output_folders]
        self.config = config
        self.lines_to_buffer = lines_to_buffer

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        with self.stats.time_stats:
            folder_pqs: list[list[HashSig]] = [[] for _ in self.data_folders] # One heap per folder
            folder_sig_readers: list[list[Iterator[HashSig]]] = [[] for _ in self.data_folders] # Readers per folder

            # Initialize readers and heaps for each data folder
            logger.info(f"Initializing readers and heaps for {len(self.data_folders)} folders for rank {rank}...")
            total_files = 0
            with ThreadPoolExecutor() as executor: # Use ThreadPool for initial reading
                init_futures = []
                for folder_idx, folder in enumerate(self.data_folders):
                    # List signature files for this specific folder and rank
                    if world_size == 1:
                        sig_files_in_folder = folder.list_files(glob_pattern="*/*" + ExtensionHelperSD.stage_1_signature)
                        if any(not sf.startswith("0000/") for sf in sig_files_in_folder):
                             logger.warning(f"world_size=1 but found sig files for different hash buckets in {folder.path}.")
                    else:
                        sig_files_in_folder = folder.list_files(
                            subdirectory=f"{rank:04d}",
                            glob_pattern="*" + ExtensionHelperSD.stage_1_signature,
                        )

                    if not sig_files_in_folder:
                        logger.debug(f"No signature files found for rank {rank} in folder {folder.path}.")
                        continue # Skip this folder if it has no relevant files

                    total_files += len(sig_files_in_folder)

                    # Create readers for this folder's files
                    current_folder_readers = [
                        read_sigs(
                            file,
                            file_i, # file_i is the index within this folder's reader list
                            self.config.hash_config,
                            lines_to_buffer=self.lines_to_buffer,
                        )
                        for file_i, file in enumerate(folder.open_files(sig_files_in_folder))
                    ]
                    folder_sig_readers[folder_idx] = current_folder_readers

                    # Asynchronously get the first element from each reader for this folder
                    folder_futures = [executor.submit(lambda r: next(r, None), reader) for reader in current_folder_readers]
                    init_futures.append((folder_idx, folder_futures)) # Store folder index and its futures

                # Collect initial elements and build heaps
                for folder_idx, futures in init_futures:
                    initial_elements = [future.result() for future in futures]
                    folder_pqs[folder_idx] = [elem for elem in initial_elements if elem is not None]
                    if folder_pqs[folder_idx]:
                        heapq.heapify(folder_pqs[folder_idx])

            logger.info(f"Initialization complete. Found {total_files} total files. Processing rank {rank}.")

            # Create a list of output managers, one for each output folder
            output_managers = [
                folder.get_output_file_manager(mode="wb") for folder in self.output_folders
            ]
            packer = struct.Struct("<I") # Packer for doc_id (uint32)

            try: # Use try...finally to ensure managers are closed
                while any(folder_pqs): # Continue as long as any heap is not empty
                    # Find the minimum hash value currently at the top of any heap
                    tops = [(pq[0], i) for i, pq in enumerate(folder_pqs) if pq]
                    if not tops:
                        break # Should not happen if any(folder_pqs) is true, but safety check

                    min_hash_value = min(top_sig.hash_value for top_sig, folder_idx in tops)

                    # Identify all folders having this minimum hash value at their top
                    min_hash_folder_indices = [
                        folder_idx for top_sig, folder_idx in tops if top_sig.hash_value == min_hash_value
                    ]

                    # Check if the count matches the required k_occurrences
                    is_target_count = len(min_hash_folder_indices) == self.config.k

                    # Process all folders that have the current minimum hash
                    for folder_idx in min_hash_folder_indices:
                        # Pop the element with the minimum hash from this folder's heap
                        popped_sig = heapq.heappop(folder_pqs[folder_idx])

                        # Write for deletion
                        if not is_target_count:
                            out_filename = f"{rank:04d}/{popped_sig.file_stem}{ExtensionHelperSD.stage_2_duplicates}"
                            doc_id_bytes = packer.pack(popped_sig.doc_id)
                            # Use the output manager corresponding to the folder index
                            output_managers[folder_idx].write(out_filename, doc_id_bytes)
                            self.stat_update(StatHints.dropped) # Count records dropped
                        else:
                            self.stat_update(StatHints.forwarded) # Count records kept

                        # Advance the reader from which the popped element came
                        # Ensure the reader list for the folder_idx is not empty before accessing
                        if folder_sig_readers[folder_idx]:
                             # Check if the file_id is valid for the current list of readers
                             if popped_sig.file_id < len(folder_sig_readers[folder_idx]):
                                 reader = folder_sig_readers[folder_idx][popped_sig.file_id]
                                 next_sig = next(reader, None)
                                 if next_sig:
                                     # Assert that the next hash is strictly greater than the previous one from the same reader
                                     # This checks for duplicates within a single input file stream
                                     assert next_sig.hash_value > popped_sig.hash_value, (
                                         f"Consecutive duplicate hash ({popped_sig.hash_value}) detected in input stream. "
                                         f"Folder index: {folder_idx}, File stem: {popped_sig.file_stem}, "
                                         f"Reader index (file_id): {popped_sig.file_id}. "
                                         f"Prev doc_id: {popped_sig.doc_id}, Next doc_id: {next_sig.doc_id}"
                                     )
                                     heapq.heappush(folder_pqs[folder_idx], next_sig)
                             else:
                                logger.error(f"Invalid file_id {popped_sig.file_id} for folder_idx {folder_idx} with {len(folder_sig_readers[folder_idx])} readers. Sig: {popped_sig}")

            finally: # Ensure all output managers are closed
                logger.info(f"Finished processing rank {rank}. Closing output managers.")
                for om in output_managers:
                    om.close()


class KeepRecordsKOccurrancesFilter(PipelineStep):
    """KeepRecordsKOccurrancesFilter: Third pipeline step
        KeepRecordsKOccurrancesFilter reads a DocumentPipeline and removes duplicated urls found at stage 2

    Args:
        data_folder: data folder to get duplicate files.
        config: config for the dedup
        exclusion_writer: writer to save excluded documents
    """

    type = "🫂 - DEDUPS"
    name = "💥 keep records k occurrences filter stage 3"

    def __init__(
        self,
        data_folder: DataFolderLike,
        config: KeepRecordsKOccurrancesConfig,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.config = config
        self.exclusion_writer = exclusion_writer

    def read_duplicates(self, file: BinaryIO, dup_dtype: np.dtype) -> np.ndarray:
        """Helper function to read duplicates from a binary file storing (doc_id) as created by the second stage."""
        with file as f:
            return read_np_from_file(f, dtype=dup_dtype, is_local_file=self.data_folder.is_local())

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        folders = self.data_folder.list_files(include_directories=True, recursive=False)
        # for performance reasons when having for instance 12k*10k files
        files = [
            f
            for f in [f"{folder}/{rank:05d}{ExtensionHelperSD.stage_2_duplicates}" for folder in folders]
            if self.data_folder.exists(f)
        ]

        logger.info(f"Loading duplicate indexes for filter from {len(files)} results files in {self.data_folder.path}.")

        dup_dtype = get_sig_dtype(self.config.hash_config)[1]
        all_dups = np.array([], dtype=dup_dtype)
        if files:
            with ThreadPoolExecutor() as pool:
                read_partial = partial(self.read_duplicates, dup_dtype=dup_dtype)
                all_dups = np.concatenate(
                    list(
                        tqdm(
                            pool.map(read_partial, self.data_folder.open_files(files)),
                            total=len(files),
                        )
                    ),
                    axis=0,
                )
            all_dups.sort()

        logger.info("Loaded duplicate indexes.")
        dups_doc_i = 0
        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
            with self.stats.time_stats:
                for doc_idx, doc in enumerate(data):
                    self.stat_update(StatHints.total)
                    with self.stats.time_stats:
                        if dups_doc_i < all_dups.shape[0] and all_dups[dups_doc_i] == doc_idx:
                            if writer:
                                writer.write(doc, rank=rank)
                            self.stat_update(StatHints.dropped)
                            dups_doc_i += 1
                        else:
                            self.stat_update(StatHints.forwarded)
                            self.update_doc_stats(doc)
                            yield doc

