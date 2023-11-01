import contextlib
import heapq
import struct
from dataclasses import dataclass
from typing import Generator

import numpy as np
from loguru import logger
from nltk import ngrams, word_tokenize

from datatrove.data import DocumentsPipeline
from datatrove.io import BaseInputDataFolder, BaseOutputDataFolder, InputDataFile
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup.utils import read_tuples_from_file, sha1_hash32, simplify_content
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.typeshelper import StatHints


# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = np.uint64((1 << 61) - 1)
_max_hash = np.uint64((1 << 32) - 1)
_hash_range = 1 << 32

"""
n_grams -> roughly nr of words (this should be small enough to catch fuzzy matches but big enough to not have each shingle be too common)
threshold is (1/14)^(1/8)~0.72
threshold is real minhash similarity cutoff for high probability inclusion by LSH minhash
probability of inclusion for s=0.8: 1-(1-0.8^8)^14=0.924
"""

DEFAULT_NR_BUCKETS = 14
DEFAULT_PER_BUCKET = 8
DEFAULT_N_GRAMS = 5

SENTINEL = (1 << 32) - 1


@dataclass
class HashSig:
    sig: tuple[int]
    doc_id: int
    file_id: int
    reader_id: int

    def to_tuple(self):
        return self.sig, self.file_id, self.doc_id, self.reader_id

    def is_from_index(self):
        return self.reader_id != self.file_id

    def __lt__(self, other):
        return self.to_tuple() < other.to_tuple()


def read_sigs(file: InputDataFile, reader_id: int, hashes_per_bucket: int, index_file: bool = False) -> Generator:
    n = hashes_per_bucket + 1 if not index_file else hashes_per_bucket
    for data in read_tuples_from_file(file, f"{n}I"):
        if index_file:
            yield HashSig(sig=data, doc_id=-1, file_id=-1, reader_id=reader_id)
        else:
            yield HashSig(sig=data[:-1], doc_id=data[-1], file_id=reader_id, reader_id=reader_id)


class MinhashDedupSignature(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 1"

    def __init__(
        self,
        output_folder: BaseOutputDataFolder,
        num_buckets: int = DEFAULT_NR_BUCKETS,
        hashes_per_bucket: int = DEFAULT_PER_BUCKET,
        n_grams: int = DEFAULT_N_GRAMS,
        seed: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_folder = output_folder
        self.n_grams = n_grams
        self.num_buckets = num_buckets
        self.hashes_per_bucket = hashes_per_bucket
        self.num_hashes = self.num_buckets * self.hashes_per_bucket
        self.seed = seed
        self._parameters = None

    @property
    def parameters(self):
        if not self._parameters:
            # Create parameters for a random bijective permutation function
            # that maps a 32-bit hash value to another 32-bit hash value.
            # http://en.wikipedia.org/wiki/Universal_hashing
            gen = np.random.RandomState(self.seed)
            self._parameters = gen.randint(
                1, _mersenne_prime, dtype=np.uint64, size=(1, self.num_hashes)
            ), gen.randint(0, _mersenne_prime, dtype=np.uint64, size=(1, self.num_hashes))
        return self._parameters

    def get_signature(self, shingles):
        a, b = self.parameters
        phv = np.bitwise_and((shingles * a + b) % _mersenne_prime, _max_hash)
        return [x.tolist() for x in np.split(np.min(phv, axis=0).astype(np.uint32), self.num_buckets)]

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def get_shingles(self, text):
        return np.array(
            [
                [sha1_hash32(" ".join(x).encode("utf-8"))]
                for x in ngrams(word_tokenize(simplify_content(text)), self.n_grams)
            ],
            dtype=np.uint64,
        )

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        buckets = [
            self.output_folder.open(f"bucket_{bi:03d}/{rank:05d}.minhash.sig", mode="wb")
            for bi in range(self.num_buckets)
        ]
        for doc_idx, doc in enumerate(data):
            self.stat_update(StatHints.total)
            shingles = self.get_shingles(doc.content)
            if shingles.size != 0:
                sig = self.get_signature(shingles)
                for bi, (bucket, bucket_sig) in enumerate(zip(buckets, sig)):
                    # print(f"{self.hashes_per_bucket=} {bucket_sig=}")
                    bucket.write(struct.pack("<%sI" % self.hashes_per_bucket, *bucket_sig))
                    bucket.write(struct.pack("<I", doc_idx))
        logger.info("Sorting buckets...")
        for bi, bucket in enumerate(buckets):
            bucket.close()
            fo = self.output_folder.open(bucket.relative_path, mode="r+b", overwrite=True)
            mmap = np.memmap(fo.file_handler, dtype=[(str(i), np.uint32) for i in range(self.hashes_per_bucket + 1)])
            mmap.sort(order=[str(i) for i in range(self.hashes_per_bucket + 1)])
            mmap.flush()
            fo.close()
        self.output_folder.close()


class MinhashDedupBuckets(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 2"

    def __init__(
        self,
        input_folder: BaseInputDataFolder,
        output_folder: BaseOutputDataFolder,
        index_folder: BaseInputDataFolder = None,
        hashes_per_bucket: int = DEFAULT_PER_BUCKET,
        num_buckets: int = DEFAULT_NR_BUCKETS,
        only_dedup_in_index: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_buckets = num_buckets
        self.hashes_per_bucket = hashes_per_bucket
        self.index_folder = index_folder
        self.only_dedup_in_index = only_dedup_in_index

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.input_folder.set_lock(dl_lock)
        self.output_folder.set_lock(up_lock)

    def __call__(self, data: DocumentsPipeline, bucket: int = 0, world_size: int = 1):
        assert data is None, "You should not use an input block before MinhashDedupBuckets"
        assert world_size == self.num_buckets, "You must run exactly one task per bucket"
        sig_files = self.input_folder.list_files(suffix=f"bucket_{bucket:03d}")
        sig_readers = [read_sigs(file, file_i, self.hashes_per_bucket) for file_i, file in enumerate(sig_files)]
        index_files = self.index_folder.list_files(suffix=f"bucket_{bucket:03d}") if self.index_folder else None
        if index_files:
            logger.info(f"Found index file(s): {', '.join([file.relative_path for file in index_files])}")
            sig_readers.extend(
                [
                    read_sigs(file, len(sig_readers) + file_i, self.hashes_per_bucket, index_file=True)
                    for file_i, file in enumerate(index_files)
                ]
            )

        pq = [next(sig_reader) for sig_reader in sig_readers]
        heapq.heapify(pq)

        out_f = self.output_folder.open(f"{bucket:05d}.dups", mode="wb")

        last: HashSig | None = None
        while pq:
            v: HashSig = heapq.heappop(pq)
            if last and last.sig == v.sig and not v.is_from_index():
                # write (file_id1, doc_id1, file_id2, doc_id2)
                if last.is_from_index():
                    # we can't actually write -1
                    out_f.write(struct.pack("<4I", SENTINEL, SENTINEL, v.file_id, v.doc_id))
                # if there isn't an index, or we are not only deduping in relation to the index
                elif not index_files or not self.only_dedup_in_index:
                    out_f.write(struct.pack("<4I", last.file_id, last.doc_id, v.file_id, v.doc_id))
            last = v
            next_sig = next(sig_readers[v.reader_id], None)
            if next_sig:
                heapq.heappush(pq, next_sig)
        self.output_folder.close()


class MinhashDedupCluster(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 3"

    def __init__(
        self,
        input_folder: BaseInputDataFolder,
        output_folder: BaseOutputDataFolder,
        num_buckets: int = DEFAULT_NR_BUCKETS,
        save_cluster_id: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_buckets = num_buckets
        self.save_cluster_id = save_cluster_id

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.input_folder.set_lock(dl_lock)
        self.output_folder.set_lock(up_lock)

    def __call__(self, data: DocumentsPipeline, _: int = 0, world_size: int = 1):
        dup_files = self.input_folder.list_files(extension=".dups")
        assert len(dup_files) == self.num_buckets, "There should be exactly one .dups file per bucket"
        assert world_size == 1, "World size must be 1 for clustering"
        union_set = {}

        def parent(x):
            if x not in union_set or union_set[x] == x:
                return x
            union_set[x] = parent(union_set[x])
            return union_set[x]

        for dup_file in dup_files:
            for f1, d1, f2, d2 in read_tuples_from_file(dup_file, "4I"):
                a, b = (f1, d1), (f2, d2)
                union_set[parent(b)] = parent(a)

        ci = 0
        cluster_ids = {}
        for node in sorted(union_set.keys()):
            file, doc = node
            p = parent(node)
            if node != p:
                self.output_folder.open(f"{file:06d}.remove", mode="wb").write(struct.pack("<I", doc))
            if self.save_cluster_id:
                if p not in cluster_ids:
                    cluster_ids[p] = ci
                    ci += 1
                self.output_folder.open(f"{file:06d}.clusters", mode="wb").write(struct.pack("<I", doc))
                self.output_folder.open(f"{file:06d}.clusters", mode="wb").write(struct.pack("<I", cluster_ids[p]))
        self.output_folder.close()


class MinhashDedupFilter(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 4"

    def __init__(
        self,
        input_folder: BaseInputDataFolder,
        exclusion_writer: DiskWriter = None,
        load_cluster_ids: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_folder = input_folder
        self.exclusion_writer = exclusion_writer
        self.load_cluster_ids = load_cluster_ids

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.data_folder.set_lock(dl_lock)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        remove_file = self.data_folder.get_file(f"{rank:06d}.remove")

        clusters_data = self.data_folder.get_files_shard(rank, world_size, extension=".clusters")
        assert (
            not self.load_cluster_ids or len(clusters_data) <= 1
        ), f"Must have exactly one .clusters file per task. Found {len(clusters_data)} files."

        if not remove_file:
            logger.warning(f"No .remove file for {rank=}.")
            for doc in data:
                self.stat_update(StatHints.total)
                self.stat_update(StatHints.forwarded)
                yield doc
            return
        with remove_file.open_binary() as f:
            with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as exc_writer:

                def get_next():
                    data = f.read(struct.calcsize("I"))
                    if data:
                        return struct.unpack("<I", data)[0]

                def load_clusters():
                    if clusters_data:
                        yield from read_tuples_from_file(clusters_data[0], "2I")

                if self.load_cluster_ids:
                    cluster_loader = load_clusters()
                    next_cluster = next(cluster_loader, None)

                next_removal = get_next()
                for idx, doc in enumerate(data):
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
        input_folder: BaseInputDataFolder,
        output_folder: BaseOutputDataFolder,
        index_name: str,
        hashes_per_bucket: int = DEFAULT_PER_BUCKET,
        num_buckets: int = DEFAULT_NR_BUCKETS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_buckets = num_buckets
        self.hashes_per_bucket = hashes_per_bucket
        self.index_name = index_name

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.input_folder.set_lock(dl_lock)
        self.output_folder.set_lock(up_lock)

    def __call__(self, data: DocumentsPipeline, bucket: int = 0, world_size: int = 1):
        assert data is None, "You should not use an input block before MinhashDedupBuckets"
        assert world_size == self.num_buckets, "You must run exactly one task per bucket"
        sig_files = self.input_folder.list_files(suffix=f"bucket_{bucket:03d}")
        sig_readers = [read_sigs(file, file_i, self.hashes_per_bucket) for file_i, file in enumerate(sig_files)]

        pq = [next(sig_reader) for sig_reader in sig_readers]
        heapq.heapify(pq)

        # writes all the sigs for the entire bucket, sequentially
        out_f = self.output_folder.open(f"bucket_{bucket:03d}/{self.index_name}.minhash.index", mode="wb")

        last: HashSig | None = None
        while pq:
            v: HashSig = heapq.heappop(pq)
            if not last or last.sig != v.sig:
                out_f.write(struct.pack("<%dI" % self.hashes_per_bucket, *v.sig))
            last = v
            next_sig = next(sig_readers[v.file_id], None)
            if next_sig:
                heapq.heappush(pq, next_sig)
        self.output_folder.close()
