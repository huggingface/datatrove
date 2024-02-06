import contextlib
import heapq
import os
import re
import struct
from dataclasses import dataclass
from functools import cache
from typing import Generator

import numpy as np
from fsspec.spec import AbstractBufferedFile
from loguru import logger

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup.utils import read_tuples_from_file, sha1_hash32, sha1_hash64, simplify_text
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.typeshelper import StatHints


# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = np.uint64((1 << 61) - 1)
_max_hash_32b = np.uint64((1 << 32) - 1)

"""
n_grams -> roughly nr of words (this should be small enough to catch fuzzy matches but big enough to not have each shingle be too common)
threshold is (1/14)^(1/8)~0.72
threshold is real minhash similarity cutoff for high probability inclusion by LSH minhash
probability of inclusion for s=0.8: 1-(1-0.8^8)^14=0.924
"""

SENTINEL = (1 << 32) - 1


@dataclass
class MinhashConfig:
    n_grams: int = 5

    num_buckets: int = 14
    hashes_per_bucket: int = 8

    use_64bit_hashes: bool = False
    seed: int = 1

    @property
    def hash_dtype(self):
        return np.uint64 if self.use_64bit_hashes else np.uint32

    @property
    def hash_format(self):
        return "Q" if self.use_64bit_hashes else "I"

    def __str__(self):
        return (
            f"{self.n_grams}ng_{self.num_buckets}bs_{self.hashes_per_bucket}hs_"
            f"{'64' if self.use_64bit_hashes else '32'}b"
        )


DEFAULT_MINHASH_CONFIG = MinhashConfig()


@dataclass(order=True)
class HashSig:
    sig: tuple[int]
    file_id: int
    doc_id: int
    reader_id: int

    def is_from_index(self):
        return self.reader_id != self.file_id


def seek_to_start(f: AbstractBufferedFile, start_hash: int, config: MinhashConfig, index_file: bool = False):
    if start_hash == 0:
        return
    line_size = struct.calcsize(f"{config.hashes_per_bucket}{config.hash_format}{'I' if not index_file else ''}")
    nr_lines = f.size // line_size

    @cache
    def read_line_start(line):
        assert line >= 0 and line < nr_lines
        f.seek(line * line_size, os.SEEK_SET)
        return struct.unpack(config.hash_format, f.read(struct.calcsize(config.hash_format)))[0]

    # save some time with binary search
    # this file is strictly bigger
    if read_line_start(0) >= start_hash:
        f.seek(0, os.SEEK_SET)
        return

    # this file is strictly smaller, ignore it completely
    if read_line_start(nr_lines - 1) < start_hash:
        f.seek(0, os.SEEK_END)
        return

    # binary search to find start line
    start_line, hi = 0, nr_lines
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    while start_line < hi:
        mid = (start_line + hi) // 2
        if read_line_start(mid) < start_hash:
            start_line = mid + 1
        else:
            hi = mid

    if start_line > nr_lines:
        raise ValueError

    # verification check. we know start_line > 0 from the check above
    if (prev_hash := read_line_start(start_line - 1)) >= start_hash:
        raise ValueError(f"Wrong bsearch start line: {prev_hash=} >= {start_hash=}")
    f.seek(start_line * line_size, os.SEEK_SET)


def read_sigs(
    file: AbstractBufferedFile,
    reader_id: int,
    config: MinhashConfig,
    index_file: bool = False,
    min_hash: int = 0,
    max_hash: int = _mersenne_prime,
    ensure_order: bool = True,
) -> Generator:
    with file as f:
        seek_to_start(f, min_hash, config, index_file)
        last = None
        for data in read_tuples_from_file(
            f, f"{config.hashes_per_bucket}{config.hash_format}{'I' if not index_file else ''}"
        ):
            sigdata = data if index_file else data[:-1]
            assert sigdata[0] >= min_hash and (
                ensure_order is False or last is None or sigdata >= last
            ), f"Hash order error. {f.tell()=}, {min_hash=}, {sigdata=}, {last=}"
            if sigdata[0] >= max_hash:
                break
            last = sigdata
            yield (
                HashSig(sig=sigdata, doc_id=-1, file_id=-1, reader_id=reader_id)
                if index_file
                else HashSig(sig=sigdata, doc_id=data[-1], file_id=reader_id, reader_id=reader_id)
            )


class MinhashDedupSignature(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 1"
    _requires_dependencies = ["nltk"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        config: MinhashConfig = DEFAULT_MINHASH_CONFIG,
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.config = config
        self.num_hashes = self.config.num_buckets * self.config.hashes_per_bucket
        self._parameters = None
        self._hash_func = sha1_hash32 if not self.config.use_64bit_hashes else sha1_hash64

    @property
    def parameters(self):
        if not self._parameters:
            # Create parameters for a random bijective permutation function
            # that maps a 32-bit hash value to another 32-bit hash value.
            # http://en.wikipedia.org/wiki/Universal_hashing
            gen = np.random.RandomState(self.config.seed)
            self._parameters = (
                gen.randint(1, _mersenne_prime, dtype=np.uint64, size=(1, self.num_hashes)),
                gen.randint(0, _mersenne_prime, dtype=np.uint64, size=(1, self.num_hashes)),
            )
        return self._parameters

    def get_signature(self, shingles):
        a, b = self.parameters
        phv = (shingles * a + b) % _mersenne_prime
        if not self.config.use_64bit_hashes:
            phv = np.bitwise_and(phv, _max_hash_32b)
        return [
            x.tolist() for x in np.split(np.min(phv, axis=0).astype(self.config.hash_dtype), self.config.num_buckets)
        ]

    def get_shingles(self, text):
        from nltk import ngrams, word_tokenize

        return np.array(
            [
                [self._hash_func(" ".join(x).encode("utf-8"))]
                for x in ngrams(word_tokenize(simplify_text(text)), self.config.n_grams)
            ],
            dtype=np.uint64,
        )

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
                                f"<{self.config.hashes_per_bucket}{self.config.hash_format}I", *bucket_sig, doc_idx
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
                    )
                )
                with self.output_folder.open(f"bucket_{bi:03d}/{rank:05d}.minhash.sig", mode="wb") as fo:
                    for sig in sigs:
                        fo.write(
                            struct.pack(
                                f"<{self.config.hashes_per_bucket}{self.config.hash_format}I", *sig.sig, sig.doc_id
                            )
                        )


class MinhashDedupBuckets(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 2"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        index_folder: DataFolderLike = None,
        config: MinhashConfig = DEFAULT_MINHASH_CONFIG,
        only_dedup_in_index: bool = True,
        create_index_name: str = None,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.index_folder = get_datafolder(index_folder) if index_folder else None
        self.config = config
        self.only_dedup_in_index = only_dedup_in_index
        self.create_index_name = create_index_name

    def get_worker_hash_range(self, sig_files, rank, world_size):
        workers_per_bucket = world_size // self.config.num_buckets
        bucket, bucket_worker = divmod(rank, workers_per_bucket)
        hash_min, hash_max = 0, _mersenne_prime if self.config.use_64bit_hashes else _max_hash_32b
        if workers_per_bucket > 1 and len(sig_files):
            # take the first file and find bucket_worker boundaries. all workers in a bucket process the same set of
            # files, so this should be consistent across workers (and span the entire range of hashes)
            with self.input_folder.open(sig_files[0], mode="rb") as f:
                line_size = struct.calcsize(f"{self.config.hashes_per_bucket}{self.config.hash_format}I")
                L, rem = divmod(f.size, line_size)
                assert rem == 0, "file size not divisible by line size"
                assert L >= workers_per_bucket, f"tried to use {workers_per_bucket=} but there are only {L} lines"
                if bucket_worker > 0:
                    # not first
                    f.seek(line_size * (L // workers_per_bucket) * bucket_worker, os.SEEK_SET)
                    hash_min = struct.unpack(
                        self.config.hash_format, f.read(struct.calcsize(self.config.hash_format))
                    )[0]
                if bucket_worker + 1 < workers_per_bucket:
                    # not last
                    f.seek(line_size * (L // workers_per_bucket) * (bucket_worker + 1), os.SEEK_SET)
                    hash_max = struct.unpack(
                        self.config.hash_format, f.read(struct.calcsize(self.config.hash_format))
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
                read_sigs(file, file_i, self.config, min_hash=hash_min, max_hash=hash_max)
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
                                out_f.write(struct.pack("<4I", SENTINEL, SENTINEL, v.file_id, v.doc_id))
                                self.stat_update("index_match", "total_matches")
                            # if there isn't an index, or we are not only deduping in relation to the index
                            elif not index_files or not self.only_dedup_in_index:
                                out_f.write(struct.pack("<4I", last.file_id, last.doc_id, v.file_id, v.doc_id))
                                self.stat_update("total_matches")
                        elif out_index:
                            # new sig that isn't part of any index, save to our new index
                            out_index.write(
                                struct.pack(f"<%d{self.config.hash_format}" % self.config.hashes_per_bucket, *v.sig)
                            )
                    last = v
                    next_sig = next(sig_readers[v.reader_id], None)
                    if next_sig:
                        assert next_sig >= v, f"Next sig sort error. {next_sig=} < {v=}"
                        heapq.heappush(pq, next_sig)
                if out_index:
                    out_index.close()


class MinhashDedupCluster(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 3"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        config: MinhashConfig = DEFAULT_MINHASH_CONFIG,
        save_cluster_id: bool = False,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.config = config
        self.save_cluster_id = save_cluster_id

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
            union_set[x] = parent(union_set[x])
            return union_set[x]

        with self.track_time():
            for dup_file in dup_files:
                with self.input_folder.open(dup_file, "rb") as dupf:
                    for f1, d1, f2, d2 in read_tuples_from_file(dupf, "4I"):
                        a, b = (f1, d1), (f2, d2)
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
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 4"

    def __init__(
        self,
        input_folder: DataFolderLike,
        exclusion_writer: DiskWriter = None,
        load_cluster_ids: bool = False,
    ):
        super().__init__()
        self.data_folder = get_datafolder(input_folder)
        self.exclusion_writer = exclusion_writer
        self.load_cluster_ids = load_cluster_ids

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
                            yield from read_tuples_from_file(clustersf, "2I")

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
    type = "🫂 - DEDUP"
    name = "🎯 MinHash build index"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        index_name: str,
        config: MinhashConfig = DEFAULT_MINHASH_CONFIG,
    ):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.config = config
        self.index_name = index_name

    def run(self, data: DocumentsPipeline = None, bucket: int = 0, world_size: int = 1):
        assert data is None, "You should not use an input block before MinhashBuildIndex"
        assert world_size == self.config.num_buckets, "You must run exactly one task per bucket"
        sig_files = self.input_folder.list_files(subdirectory=f"bucket_{bucket:03d}")
        sig_readers = [
            read_sigs(file, file_i, self.config)
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
                    out_f.write(struct.pack(f"<%d{self.config.hash_format}" % self.config.hashes_per_bucket, *v.sig))
                last = v
                next_sig = next(sig_readers[v.file_id], None)
                if next_sig:
                    heapq.heappush(pq, next_sig)
        out_f.close()
