"""
URL based deduplication.
"""

import contextlib
import heapq
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import BinaryIO, Callable, Generator

import numpy as np
from fsspec.spec import AbstractBufferedFile
from loguru import logger
from tqdm import tqdm

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.binaryio import read_np_from_file, read_tuples_from_file
from datatrove.utils.text import xxhash64
from datatrove.utils.typeshelper import ExtensionHelperSD, StatHints

from ..writers.disk_base import DiskWriter


@dataclass
class UrlDedupConfig:
    """
    Args:
        url_normalizer: Callable[[str], str] Normalize the url, e.g. remove query parameters
        document_priority: Callable[[Document], int] Function for determining the priority of a document.
            Only the document with the highest priority will be preserved, out of duplicates.
            The document priority must be in range [1, 65535]
    """

    url_normalizer: Callable[[str], str] | None = None
    document_priority: Callable[[Document], int] | None = None


DEFAULT_URL_DEDUP_CONFIG = UrlDedupConfig()


@dataclass(order=False)
class HashSig:
    hash_value: int
    priority: int
    doc_id: int
    file_id: int

    def is_from_index(self):
        return self.doc_id == -1 and self.priority == 1

    def __lt__(self, other: "HashSig") -> bool:
        # Ensure that highest priority is always first of the hashes
        return (self.hash_value, -self.priority, self.doc_id) < (
            other.hash_value,
            -other.priority,
            other.doc_id,
        )


SIG_DTYPE = np.dtype([("hash", "<u8"), ("priority", "<u2"), ("doc", "<u4")])
DUP_DTYPE = SIG_DTYPE[2]


class UrlDedupSignature(PipelineStep):
    """UrlDedup: First pipeline step
        Creates a signature for url in each document. Each HashSig has n hash, the priority the doc id. Before saving
        them the hashes are sorted based on (hash, -priority, doc_id).

    Args:
        output_folder: folder where signatures are saved
        finder_workers: number of workers used in finder stage of deduplication
        config: configuration for the dedup
    """

    type = "🫂 - DEDUPS"
    name = "💥 url-deduplication stage 1"
    _requires_dependencies = ["xxhash"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        finder_workers: int = 1,
        config: UrlDedupConfig = DEFAULT_URL_DEDUP_CONFIG,
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        if finder_workers <= 0:
            raise ValueError("finder_workers must be >= 1")
        elif finder_workers > 1:
            logger.warning(f"Remember to also set the number of tasks of the finder block to {finder_workers=}!")
        self.finder_workers = finder_workers
        self.config = config

    def save_hashes(self, rank: int, signatures):
        priority_max = np.iinfo(SIG_DTYPE["priority"]).max

        # 0 will stay as is, so we can't use 0 as a priority
        assert all(
            sig[1] >= 1 and sig[1] <= priority_max for sig in signatures
        ), f"priority must be between 1 and {priority_max}"
        signatures = np.array(signatures, dtype=SIG_DTYPE)

        # Ensure that the highest priority is always first
        signatures["priority"] = -signatures["priority"]
        signatures.sort(axis=0)
        signatures["priority"] = -signatures["priority"]

        # Same code as in sentence_dedup
        hashes_per_worker = np.iinfo(np.uint64).max // self.finder_workers
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
        normalized_url: str = (
            self.config.url_normalizer(doc.metadata["url"]) if self.config.url_normalizer else doc.metadata["url"]
        )
        priority = self.config.document_priority(doc) if self.config.document_priority else 1
        # We use xxhash as it's the fastest and we don't need cryptographic security
        hashes = [(xxhash64(normalized_url), priority, doc_idx)]

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
    index_file: bool = False,
    lines_to_buffer: int = 5,
) -> Generator[HashSig, None, None]:
    last = None
    line_format = "QHI" if not index_file else "Q"
    with file as f:
        for data in read_tuples_from_file(f, line_format, lines_to_buffer=lines_to_buffer):
            assert last is None or data[0] >= last, f"Hash order error. {f.tell()=}, {data[0]=}, {last=}"
            last = data[0]
            yield (
                HashSig(hash_value=data[0], doc_id=-1, file_id=file_id, priority=-1)
                if index_file
                else HashSig(
                    file_id=file_id,
                    hash_value=data[0],
                    priority=data[1],
                    doc_id=data[2],
                )
            )


class UrlFindDedups(PipelineStep):
    """UrlDedup: Second pipeline step
        UrlFindDedups reads all the signatures from the previous step and loads them
        in a priority queue to check for duplicates. If a duplicate is found its document id is saved.
        The document with the highest priority is the one that will be saved out of the duplicates .

    Args:
        data_folder: data folder where signatures are saved
        output_folder: folder where duplicates are saved
        index_folder: folder where index files are saved
        config: configuration for the dedup
        lines_to_buffer: number of lines to buffer (speed up reading)
    """

    type = "🫂 - DEDUPS"
    name = "💥 url-deduplication stage 2"

    def __init__(
        self,
        data_folder: DataFolderLike,
        output_folder: DataFolderLike,
        index_folder: DataFolderLike = None,
        config: UrlDedupConfig = DEFAULT_URL_DEDUP_CONFIG,
        lines_to_buffer: int = 5,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.output_folder = get_datafolder(output_folder)
        self.index_folder = get_datafolder(index_folder) if index_folder else None

        self.config = config
        self.lines_to_buffer = lines_to_buffer

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        with self.stats.time_stats:
            if world_size == 1:
                # check that there was not a mistake in setting this values
                sig_files = self.data_folder.list_files(glob_pattern="*/*" + ExtensionHelperSD.stage_1_signature)
                if any(not sig_file.startswith("0000/") for sig_file in sig_files):
                    raise ValueError(
                        f"{world_size=} but found sig files for different hash buckets. Set tasks=finder_workers"
                    )
            else:
                sig_files = self.data_folder.list_files(
                    subdirectory=f"{rank:04d}",
                    glob_pattern=ExtensionHelperSD.stage_1_signature,
                )
            sig_readers = [
                read_sigs(
                    file,
                    file_i,
                    lines_to_buffer=self.lines_to_buffer,
                )
                for file_i, file in enumerate(self.data_folder.open_files(sig_files))
            ]
            index_files = self.index_folder.list_files() if self.index_folder else None
            if index_files:
                logger.info(f"Found index file(s): {', '.join(index_files)}")
                sig_readers.extend(
                    [
                        read_sigs(
                            file,
                            len(sig_readers) + file_i,
                            index_file=True,
                            lines_to_buffer=self.lines_to_buffer,
                        )
                        for file_i, file in enumerate(self.data_folder.open_files(index_files))
                    ]
                )

            logger.info(f"Initializing pq with {len(sig_readers)} files.")
            with ThreadPoolExecutor() as executor:
                pq = [
                    x
                    for x in tqdm(
                        executor.map(lambda x: next(x, None), sig_readers),
                        total=len(sig_readers),
                        desc="Initializing pq...",
                    )
                    if x
                ]
            heapq.heapify(pq)
            logger.info("PQ initialized.")

            output_mg = self.output_folder.get_output_file_manager(mode="wb")
            last: HashSig | None = None
            packer = struct.Struct("<I")
            while pq:
                v: HashSig = heapq.heappop(pq)
                if last and last.hash_value == v.hash_value and not v.is_from_index():
                    out_filename = f"{rank:04d}/{v.file_id:05d}{ExtensionHelperSD.stage_2_duplicates}"
                    doc_id_bytes = packer.pack(v.doc_id)
                    output_mg.write(out_filename, doc_id_bytes)
                last = v
                new_v = next(sig_readers[v.file_id], None)

                if new_v:
                    heapq.heappush(pq, new_v)

        output_mg.close()


class UrlDedupFilter(PipelineStep):
    """UrlDedup: Third pipeline step
        UrlDedupFilter reads a DocumentPipeline and removes duplicated urls found at stage 2

    Args:
        data_folder: data folder to get duplicate files.
        config: config for the dedup
        exclusion_writer: writer to save excluded documents
    """

    type = "🫂 - DEDUPS"
    name = "💥 url-deduplication stage 3"

    def __init__(
        self,
        data_folder: DataFolderLike,
        config: UrlDedupConfig = DEFAULT_URL_DEDUP_CONFIG,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.config = config
        self.exclusion_writer = exclusion_writer

    def read_duplicates(self, file: BinaryIO) -> np.ndarray:
        """Helper function to read duplicates from a binary file storing (doc_id) as created by the second stage."""
        with file as f:
            return read_np_from_file(f, dtype=DUP_DTYPE, is_local_file=self.data_folder.is_local())

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        folders = self.data_folder.list_files(include_directories=True, recursive=False)
        # for performance reasons when having for instance 12k*10k files
        files = [
            f
            for f in [f"{folder}/{rank:05d}{ExtensionHelperSD.stage_2_duplicates}" for folder in folders]
            if self.data_folder.exists(f)
        ]

        logger.info(f"Loading duplicate indexes from {len(files)} results files.")

        all_dups = np.array([], dtype=DUP_DTYPE)
        if files:
            with ThreadPoolExecutor() as pool:
                all_dups = np.concatenate(
                    list(
                        tqdm(
                            pool.map(self.read_duplicates, self.data_folder.open_files(files)),
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
