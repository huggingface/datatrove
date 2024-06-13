import contextlib
import heapq
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np
from fsspec.spec import AbstractBufferedFile

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.binaryio import read_tuples_from_file, seek_to_start
from datatrove.utils.hashing import HashConfig, create_hash_func
from datatrove.utils.logging import logger
from datatrove.utils.text import TextNormConfig, ngrams, simplify_text
from datatrove.utils.typeshelper import Languages, StatHints
from datatrove.utils.word_tokenizers import load_word_tokenizer


# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = np.uint64((1 << 61) - 1)

"""
n_grams -> roughly nr of words (this should be small enough to catch fuzzy matches but big enough to not have each shingle be too common)
threshold is (1/14)^(1/8)~0.72
threshold is real minhash similarity cutoff for high probability inclusion by LSH minhash
probability of inclusion for s=0.8: 1-(1-0.8^8)^14=0.924
"""

SENTINEL = (1 << 32) - 1


@dataclass
class MinhashConfig:
    """Configuration for Min-Hash deduplication

    Args:
        n_grams: n-grams size to use
        num_buckets: number of buckets to use
        hashes_per_bucket: number of hashes per bucket
        seed: random seed used to generate the hash function parameters. Should be the same on all workers to ensure they all have the same parameters
    """

    n_grams: int = 5
    num_buckets: int = 14
    hashes_per_bucket: int = 8
    seed: int = 1

    norm_config: TextNormConfig = field(default_factory=TextNormConfig)
    hash_config: HashConfig = field(default_factory=HashConfig)

    def __str__(self):
        return f"{self.n_grams}ng_{self.num_buckets}bs_{self.hashes_per_bucket}hs_{self.hash_config}"


@dataclass(order=True)
class HashSig:
    """Hash signature for a given document in a given bucket

    Args:
        sig: tuple of hashes
        file_id: file id
        doc_id: document id
        reader_id: reader id. Used to know from where the next signature should be requested
    """

    sig: tuple[int]
    file_id: int
    file_stem: str
    doc_id: int
    reader_id: int

    def is_from_index(self):
        return self.reader_id != self.file_id


def read_sigs(
    file: AbstractBufferedFile,
    reader_id: int,
    config: MinhashConfig,
    index_file: bool = False,
    min_hash: int = 0,
    max_hash: int = _mersenne_prime,
    ensure_order: bool = True,
    lines_to_buffer: int = 5,
) -> Generator:
    """Read signatures from a file

    Args:
        file: file to read from
        reader_id: reader id
        config: minhash configuration (a MinhashConfig object)
        index_file: is index file
    """
    line_format = f"{config.hashes_per_bucket}{config.hash_config.struct_format}{'I' if not index_file else ''}"
    with file as f:
        if f.size == 0:
            return
        seek_to_start(f, min_hash, line_format, config.hash_config.struct_format)
        last = None
        file_stem = Path(file.path).name.removesuffix(".minhash.sig")
        for data in read_tuples_from_file(f, line_format, lines_to_buffer=lines_to_buffer):
            sigdata = data if index_file else data[:-1]
            assert sigdata[0] >= min_hash and (
                ensure_order is False or last is None or sigdata >= last
            ), f"Hash order error. {f.tell()=}, {min_hash=}, {sigdata=}, {last=}"
            if sigdata[0] >= max_hash:
                break
            last = sigdata
            yield (
                HashSig(sig=sigdata, doc_id=-1, file_id=-1, reader_id=reader_id, file_stem=file_stem)
                if index_file
                else HashSig(sig=sigdata, doc_id=data[-1], file_id=reader_id, reader_id=reader_id, file_stem=file_stem)
            )


class MinhashDedupSignature(PipelineStep):
    """Minhash Deduplication: First Pipeline Step

        Compute the minhash signature for each document and write it to disk.

    Args:
        output_folder: output folder
        config: minhash configuration (a MinhashConfig object)
    """

    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 1"

    def __init__(self, output_folder: DataFolderLike, config: MinhashConfig = None, language: str = Languages.english):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.config = config or MinhashConfig()
        self.num_hashes = self.config.num_buckets * self.config.hashes_per_bucket
        self._parameters = None
        self._hash_func = create_hash_func(self.config.hash_config)
        self.language = language
        self.word_tokenizer = load_word_tokenizer(language)

    @property
    def parameters(self):
        """Minhash parameters

        Create parameters for a random bijective permutation function
        that maps a 32/64-bit hash value to another 32/64-bit hash value.
        http://en.wikipedia.org/wiki/Universal_hashing

        Note: For 64-bit hashes the upper-bound for codomain is not [0,2**64) but [0,2**61 - 1)
        """
        if not self._parameters:
            gen = np.random.RandomState(self.config.seed)
            self._parameters = (
                gen.randint(1, _mersenne_prime, dtype=np.uint64, size=(1, self.num_hashes)),
                gen.randint(0, _mersenne_prime, dtype=np.uint64, size=(1, self.num_hashes)),
            )
        return self._parameters

    def get_signature(self, shingles: np.ndarray) -> list[list[int]]:
        """Get the signature for a set of shingles (n-grams)

        Args:
            shingles: shingles (n-grams) numpy uint64 array of size (N, 1)

        Returns:
            list (num buckets) of lists of integers (hashes)
        """
        a, b = self.parameters
        phv = (shingles * a + b) % _mersenne_prime
        if self.config.hash_config.precision == 32:
            phv = np.bitwise_and(phv, self.config.hash_config.max)
        return [
            x.tolist()
            for x in np.split(np.min(phv, axis=0).astype(self.config.hash_config.np_dtype), self.config.num_buckets)
        ]

    def get_shingles(self, text: str) -> np.ndarray:
        """Get shingles (hashed n-grams) from a string of text

        Shingles are created by hashing n-grams of simplified text (lower cases, whitespace normalized, no punctuation, etc).

        Args:
            text: input text

        Returns:
            numpy array of shingles: dtype = uint64, shape = (number of n_grams in string, 1)
        """
        return np.fromiter(
            [
                self._hash_func(" ".join(x))
                for x in ngrams(
                    self.word_tokenizer.word_tokenize(simplify_text(text, self.config.norm_config)),
                    self.config.n_grams,
                )
            ],
            dtype=np.uint64,
        ).reshape((-1, 1))

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        buckets = [
            self.output_folder.open(f"bucket_{bi:03d}/{rank:05d}.minhash.sig", mode="wb")
            for bi in range(self.config.num_buckets)
        ]
        with self.track_time():
            for doc_idx, doc in enumerate(data):
                self.stat_update(StatHints.total)
                shingles = self.get_shingles(doc.text)
                if shingles.size != 0:
                    sig = self.get_signature(shingles)
                    for bi, (bucket, bucket_sig) in enumerate(zip(buckets, sig)):
                        # print(f"{self.hashes_per_bucket=} {bucket_sig=}")
                        bucket.write(
                            struct.pack(
                                f"<{self.config.hashes_per_bucket}{self.config.hash_config.struct_format}I",
                                *bucket_sig,
                                doc_idx,
                            )
                        )
            # TODO: prevent these files from being uploaded/redownloaded in the first place
            for file in buckets:
                file.close()

            logger.info("Sorting buckets...")
            for bi in range(len(buckets)):
                # read one by one, sort and write back
                sigs = sorted(
                    read_sigs(
                        self.output_folder.open(f"bucket_{bi:03d}/{rank:05d}.minhash.sig", mode="rb"),
                        -1,
                        self.config,
                        ensure_order=False,
                        lines_to_buffer=-1,  # load everything in one go
                    )
                )
                with self.output_folder.open(f"bucket_{bi:03d}/{rank:05d}.minhash.sig", mode="wb") as fo:
                    for sig in sigs:
                        fo.write(
                            struct.pack(
                                f"<{self.config.hashes_per_bucket}{self.config.hash_config.struct_format}I",
                                *sig.sig,
                                sig.doc_id,
                            )
                        )


class MinhashDedupBuckets(PipelineStep):
    """Minhash Deduplication: Second Pipeline Step

        Find duplicate pairs from the signatures and possibly an index. Can also save an index with the new signatures.

    Args:
        input_folder: input folder containing the signature from step 1
        output_folder: output folder where results (document duplicate pairs) will be saved
        index_folder: index folder. If set, we will load all index files in this folder and use them as a reference for deduplicating the current dataset (remove any matches on our dataset with signatures from the index)
        config: minhash configuration (a MinhashConfig object)
        only_dedup_in_index: only deduplicate versus index (ignore any matches between 2 documents in our input dataset)
        create_index_name: create index name. If this parameter is set, index files will be created with this name that other datasets can use as a reference for dedup. Set to `None` to disable index file creation.
    """

    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 2"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        index_folder: DataFolderLike = None,
        config: MinhashConfig = None,
        only_dedup_in_index: bool = True,
        create_index_name: str = None,
        lines_to_buffer: int = 5,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.index_folder = get_datafolder(index_folder) if index_folder else None
        self.config = config or MinhashConfig()
        self.only_dedup_in_index = only_dedup_in_index
        self.create_index_name = create_index_name
        self.lines_to_buffer = lines_to_buffer

    def get_worker_hash_range(self, sig_files, rank, world_size):
        workers_per_bucket = world_size // self.config.num_buckets
        bucket, bucket_worker = divmod(rank, workers_per_bucket)
        hash_min, hash_max = (
            0,
            _mersenne_prime if self.config.hash_config.precision == 64 else self.config.hash_config.max,
        )
        if workers_per_bucket > 1 and len(sig_files):
            # take the first file and find bucket_worker boundaries. all workers in a bucket process the same set of
            # files, so this should be consistent across workers (and span the entire range of hashes)
            with self.input_folder.open(sig_files[0], mode="rb") as f:
                line_size = struct.calcsize(f"{self.config.hashes_per_bucket}{self.config.hash_config.struct_format}I")
                L, rem = divmod(f.size, line_size)
                assert rem == 0, "file size not divisible by line size"
                assert L >= workers_per_bucket, f"tried to use {workers_per_bucket=} but there are only {L} lines"
                if bucket_worker > 0:
                    # not first
                    f.seek(line_size * (L // workers_per_bucket) * bucket_worker, os.SEEK_SET)
                    hash_min = struct.unpack(
                        self.config.hash_config.struct_format,
                        f.read(struct.calcsize(self.config.hash_config.struct_format)),
                    )[0]
                if bucket_worker + 1 < workers_per_bucket:
                    # not last
                    f.seek(line_size * (L // workers_per_bucket) * (bucket_worker + 1), os.SEEK_SET)
                    hash_max = struct.unpack(
                        self.config.hash_config.struct_format,
                        f.read(struct.calcsize(self.config.hash_config.struct_format)),
                    )[0]
        return hash_min, hash_max

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        assert data is None, "You should not use an input block before MinhashDedupBuckets"
        assert (world_size % self.config.num_buckets) == 0, "Number of tasks must be divisible by num_buckets"
        workers_per_bucket = world_size // self.config.num_buckets
        bucket, bucket_worker = divmod(rank, workers_per_bucket)

        with self.track_time():
            sig_files = self.input_folder.list_files(subdirectory=f"bucket_{bucket:03d}")
            hash_min, hash_max = self.get_worker_hash_range(sig_files, rank, world_size)

            logger.info(
                f"Running worker {bucket_worker + 1}/{workers_per_bucket} on bucket {bucket:03d}. "
                f"Hash range: {[hash_min, hash_max]}"
            )

            sig_readers = [
                read_sigs(
                    file,
                    file_i,
                    self.config,
                    min_hash=hash_min,
                    max_hash=hash_max,
                    lines_to_buffer=self.lines_to_buffer,
                )
                for file_i, file in enumerate(self.input_folder.open_files(sig_files, mode="rb"))
            ]

            own_index_regex = re.compile(rf"bucket_{bucket:03d}/{self.create_index_name}_\d{{2}}.minhash.index")
            index_files = (
                [
                    filename
                    for filename in self.index_folder.list_files(subdirectory=f"bucket_{bucket:03d}")
                    # exclude "itself" if the index was partially uploaded/ended midway + other workers
                    if not self.create_index_name or not own_index_regex.fullmatch(filename)
                ]
                if self.index_folder
                else None
            )
            if index_files:
                logger.info(f"Found {len(index_files)} index file(s): {', '.join(index_files)}")
                sig_readers.extend(
                    [
                        read_sigs(
                            file,
                            len(sig_readers) + file_i,
                            self.config,
                            index_file=True,
                            min_hash=hash_min,
                            max_hash=hash_max,
                            lines_to_buffer=self.lines_to_buffer,
                        )
                        for file_i, file in enumerate(self.index_folder.open_files(index_files, mode="rb"))
                    ]
                )

            pq = [x for x in [next(sig_reader, None) for sig_reader in sig_readers] if x is not None]
            heapq.heapify(pq)
            logger.info("Finished initializing signatures priority queue.")

            # out index file
            out_index = None
            if self.index_folder and self.create_index_name:
                out_index = self.index_folder.open(
                    f"bucket_{bucket:03d}/{self.create_index_name}_{bucket_worker:02d}.minhash.index", mode="wb"
                )

            with self.output_folder.open(f"{bucket:05d}_{bucket_worker:02d}.dups", mode="wb") as out_f:
                last: HashSig | None = None
                while pq:
                    v: HashSig = heapq.heappop(pq)
                    assert last is None or v >= last, f"Sig queue sort error. {v=} < {last=}"
                    if not v.is_from_index():
                        if last and last.sig == v.sig:
                            # write (file_id1, doc_id1, file_id2, doc_id2)
                            if last.is_from_index():
                                # we can't actually write -1, so we use SENTINEL instead
                                out_f.write(struct.pack("<4I", SENTINEL, SENTINEL, int(v.file_stem), v.doc_id))
                                self.stat_update("index_match", "total_matches")
                            # if there isn't an index, or we are not only deduping in relation to the index
                            elif not index_files or not self.only_dedup_in_index:
                                out_f.write(
                                    struct.pack("<4I", int(last.file_stem), last.doc_id, int(v.file_stem), v.doc_id)
                                )
                                self.stat_update("total_matches")
                        elif out_index:
                            # new sig that isn't part of any index, save to our new index
                            out_index.write(
                                struct.pack(
                                    f"<%d{self.config.hash_config.struct_format}" % self.config.hashes_per_bucket,
                                    *v.sig,
                                )
                            )
                    last = v
                    next_sig = next(sig_readers[v.reader_id], None)
                    if next_sig:
                        assert next_sig >= v, f"Next sig sort error. {next_sig=} < {v=}"
                        heapq.heappush(pq, next_sig)
                if out_index:
                    out_index.close()


class MinhashDedupCluster(PipelineStep):
    """Minhash Deduplication: Third Pipeline Step

    Cluster the documents using the previously found duplicate pairs. If A-B and B-C are duplicate pairs, then we will have the A-B-C cluster. Only one document per cluster will be kept after filtering
    """

    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 3"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        config: MinhashConfig = None,
        save_cluster_id: bool = False,
        ignore_index_matches: bool = False,
        lines_to_buffer: int = 5,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.config = config or MinhashConfig()
        self.save_cluster_id = save_cluster_id
        self.ignore_index_matches = ignore_index_matches
        self.lines_to_buffer = lines_to_buffer

    def run(self, data: DocumentsPipeline = None, _: int = 0, world_size: int = 1):
        dup_files = self.input_folder.list_files(glob_pattern="*.dups")
        assert (
            len(dup_files) % self.config.num_buckets
        ) == 0, "Number of .dups files should be divisible by number of buckets"
        assert world_size == 1, "World size must be 1 for clustering"
        union_set = {}

        def parent(x):
            if x not in union_set or union_set[x] == x:
                return x
            # Path Compression
            union_set[x] = parent(union_set[x])
            return union_set[x]

        with self.track_time():
            for dup_file in dup_files:
                with self.input_folder.open(dup_file, "rb") as dupf:
                    for f1, d1, f2, d2 in read_tuples_from_file(dupf, "4I", lines_to_buffer=self.lines_to_buffer):
                        a, b = (f1, d1), (f2, d2)
                        if self.ignore_index_matches and a == (SENTINEL, SENTINEL):
                            # if we are skipping matches with the index and "a" is from the index
                            continue
                        union_set[parent(b)] = parent(a)

            ci = 0
            cluster_ids = {}
            with self.output_folder.get_output_file_manager(mode="wb") as output_mg:
                for node in sorted(union_set.keys()):
                    self.stat_update("duplicates")
                    file, doc = node
                    p = parent(node)
                    if node != p:
                        output_mg.write(f"{file:06d}.remove", struct.pack("<I", doc))
                        self.stat_update("to_remove")
                    if self.save_cluster_id:
                        if p not in cluster_ids:
                            cluster_ids[p] = ci
                            ci += 1
                            self.stat_update("clusters")
                        output_mg.write(f"{file:06d}.clusters", struct.pack("<I", doc))
                        output_mg.write(f"{file:06d}.clusters", struct.pack("<I", cluster_ids[p]))


class MinhashDedupFilter(PipelineStep):
    """Minhash Deduplication: Fourth (and final) Pipeline Step

    Filter the documents based on the minhash clusters to keep only one per cluster
    """

    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 4"

    def __init__(
        self,
        input_folder: DataFolderLike,
        exclusion_writer: DiskWriter = None,
        load_cluster_ids: bool = False,
        lines_to_buffer: int = 5,
    ):
        super().__init__()
        self.data_folder = get_datafolder(input_folder)
        self.exclusion_writer = exclusion_writer
        self.load_cluster_ids = load_cluster_ids
        self.lines_to_buffer = lines_to_buffer

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        clusters_data = self.data_folder.get_shard(rank, world_size, glob_pattern="*.clusters")
        assert (
            not self.load_cluster_ids or len(clusters_data) <= 1
        ), f"Must have exactly one .clusters file per task. Found {len(clusters_data)} files."

        if not self.data_folder.isfile(f"{rank:06d}.remove"):
            logger.warning(f"No .remove file for {rank=}.")
            for doc in data:
                self.stat_update(StatHints.total, StatHints.forwarded)
                yield doc
            return
        with self.data_folder.open(f"{rank:06d}.remove", "rb") as f:
            with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as exc_writer:

                def get_next():
                    data = f.read(struct.calcsize("I"))
                    if data:
                        return struct.unpack("<I", data)[0]

                def load_clusters():
                    if clusters_data:
                        with self.data_folder.open(clusters_data[0], "rb") as clustersf:
                            yield from read_tuples_from_file(clustersf, "2I", lines_to_buffer=self.lines_to_buffer)

                if self.load_cluster_ids:
                    cluster_loader = load_clusters()
                    next_cluster = next(cluster_loader, None)

                next_removal = get_next()
                for idx, doc in enumerate(data):
                    with self.track_time():
                        if self.load_cluster_ids:
                            if next_cluster and idx == next_cluster[0]:
                                doc.metadata["minhash_cluster"] = next_cluster[1]
                                next_cluster = next(cluster_loader, None)

                        self.stat_update(StatHints.total)
                        if next_removal == idx:
                            # to remove
                            self.stat_update(StatHints.dropped)
                            if self.exclusion_writer:
                                exc_writer.write(doc, rank)
                            next_removal = get_next()
                            continue
                        self.stat_update(StatHints.forwarded)
                    yield doc


class MinhashBuildIndex(PipelineStep):
    """Minhash Deduplication

    Only build an index from the signatures, without deduplicating
    """

    type = "🫂 - DEDUP"
    name = "🎯 MinHash build index"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        index_name: str,
        config: MinhashConfig = None,
        lines_to_buffer: int = 5,
    ):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.config = config or MinhashConfig()
        self.index_name = index_name
        self.lines_to_buffer = lines_to_buffer

    def run(self, data: DocumentsPipeline = None, bucket: int = 0, world_size: int = 1):
        assert data is None, "You should not use an input block before MinhashBuildIndex"
        assert world_size == self.config.num_buckets, "You must run exactly one task per bucket"
        sig_files = self.input_folder.list_files(subdirectory=f"bucket_{bucket:03d}")
        sig_readers = [
            read_sigs(file, file_i, self.config, lines_to_buffer=self.lines_to_buffer)
            for file_i, file in enumerate(self.input_folder.open_files(sig_files, mode="rb"))
        ]

        pq = [next(sig_reader) for sig_reader in sig_readers]
        heapq.heapify(pq)

        # writes all the sigs for the entire bucket, sequentially
        out_f = self.output_folder.open(f"bucket_{bucket:03d}/{self.index_name}.minhash.index", mode="wb")

        last: HashSig | None = None
        with self.track_time():
            while pq:
                v: HashSig = heapq.heappop(pq)
                if not last or last.sig != v.sig:
                    out_f.write(
                        struct.pack(
                            f"<%d{self.config.hash_config.struct_format}" % self.config.hashes_per_bucket, *v.sig
                        )
                    )
                last = v
                next_sig = next(sig_readers[v.file_id], None)
                if next_sig:
                    heapq.heappush(pq, next_sig)
        out_f.close()
