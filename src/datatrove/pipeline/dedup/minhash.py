import contextlib
import heapq
import struct
from dataclasses import dataclass
from typing import BinaryIO, Generator

import numpy as np
from loguru import logger
from nltk import ngrams, word_tokenize

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup.utils import read_tuples_from_file, sha1_hash32, sha1_hash64, simplify_content
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


def read_sigs(file: BinaryIO, reader_id: int, config: MinhashConfig, index_file: bool = False) -> Generator:
    if index_file:
        for data in read_tuples_from_file(file, f"{config.hashes_per_bucket}{config.hash_format}"):
            yield HashSig(sig=data, doc_id=-1, file_id=-1, reader_id=reader_id)
    else:
        for data in read_tuples_from_file(file, f"{config.hashes_per_bucket}{config.hash_format}", "I"):
            yield HashSig(sig=data[:-1], doc_id=data[-1], file_id=reader_id, reader_id=reader_id)


class MinhashDedupSignature(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 1"

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
        return np.array(
            [
                [self._hash_func(" ".join(x).encode("utf-8"))]
                for x in ngrams(word_tokenize(simplify_content(text)), self.config.n_grams)
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
                shingles = self.get_shingles(doc.content)
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
                        self.output_folder.open(f"bucket_{bi:03d}/{rank:05d}.minhash.sig", mode="rb"), -1, self.config
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

    def run(self, data: DocumentsPipeline = None, bucket: int = 0, world_size: int = 1):
        assert data is None, "You should not use an input block before MinhashDedupBuckets"
        assert world_size == self.config.num_buckets, "You must run exactly one task per bucket"
        sig_files = self.input_folder.list_files(subdirectory=f"bucket_{bucket:03d}")
        sig_readers = [
            read_sigs(file, file_i, self.config)
            for file_i, file in enumerate(self.input_folder.open_files(sig_files, mode="rb"))
        ]
        index_files = self.index_folder.list_files(subdirectory=f"bucket_{bucket:03d}") if self.index_folder else None
        if index_files:
            logger.info(f"Found index file(s): {', '.join(index_files)}")
            sig_readers.extend(
                [
                    read_sigs(file, len(sig_readers) + file_i, self.config, index_file=True)
                    for file_i, file in enumerate(self.index_folder.open_files(index_files, mode="rb"))
                    # exclude "itself" if the index was partially uploaded/ended midway
                    if not self.create_index_name
                    or file != f"bucket_{bucket:03d}/{self.create_index_name}.minhash.index"
                ]
            )

        pq = [next(sig_reader) for sig_reader in sig_readers]
        heapq.heapify(pq)

        # out index file
        out_index = None
        if self.index_folder and self.create_index_name:
            out_index = self.index_folder.open(
                f"bucket_{bucket:03d}/{self.create_index_name}.minhash.index", mode="wb"
            )

        with self.output_folder.open(f"{bucket:05d}.dups", mode="wb") as out_f:
            last: HashSig | None = None
            with self.track_time():
                while pq:
                    v: HashSig = heapq.heappop(pq)
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
        dup_files = self.input_folder.list_files(extension=".dups")
        assert len(dup_files) == self.config.num_buckets, "There should be exactly one .dups file per bucket"
        assert world_size == 1, "World size must be 1 for clustering"
        union_set = {}

        def parent(x):
            if x not in union_set or union_set[x] == x:
                return x
            union_set[x] = parent(union_set[x])
            return union_set[x]

        with self.track_time():
            for dup_file in dup_files:
                for f1, d1, f2, d2 in read_tuples_from_file(self.input_folder.open(dup_file, "rb"), "4I"):
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
        clusters_data = self.data_folder.get_shard(rank, world_size, extension=".clusters")
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
