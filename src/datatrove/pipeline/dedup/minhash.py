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
from datatrove.pipeline.dedup.utils import sha1_hash32, simplify_content
from datatrove.utils.typeshelper import StatHints


# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = np.uint64((1 << 61) - 1)
_max_hash = np.uint64((1 << 32) - 1)
_hash_range = 1 << 32

DEFAULT_NR_BUCKETS = 20
DEFAULT_PER_BUCKET = 20
DEFAULT_N_GRAMS = 5


@dataclass
class HashSig:
    sig: list[int]
    doc_id: int
    file_id: int

    def to_tuple(self):
        return self.sig, self.file_id, self.doc_id

    def __lt__(self, other):
        return self.to_tuple() < other.to_tuple()


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

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        buckets = [
            self.output_folder.open(f"bucket_{bi:03d}/{rank:05d}.minhash.sig", mode="wb")
            for bi in range(self.num_buckets)
        ]
        for doc_idx, doc in enumerate(data):
            self.stat_update(StatHints.total)
            shingles = np.array(
                [
                    [sha1_hash32(" ".join(x).encode("utf-8"))]
                    for x in ngrams(word_tokenize(simplify_content(doc.content)), self.n_grams)
                ],
                dtype=np.uint64,
            )
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
        hashes_per_bucket: int = DEFAULT_PER_BUCKET,
        num_buckets: int = DEFAULT_NR_BUCKETS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_buckets = num_buckets
        self.hashes_per_bucket = hashes_per_bucket

    def read_sigs(self, file: InputDataFile, file_id: int) -> Generator:
        n = self.hashes_per_bucket + 1
        with file.open(binary=True) as f:
            while True:
                data = f.read(n * 4)
                if not data:
                    return
                data = struct.unpack("<%sI" % n, n)[0]
                yield HashSig(sig=data[:-1], doc_id=data[-1], file_id=file_id)

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.input_folder.set_lock(dl_lock)
        self.output_folder.set_lock(up_lock)

    def __call__(self, data: DocumentsPipeline, bucket: int = 0, world_size: int = 1):
        assert world_size == self.num_buckets, "You must run exactly one task per bucket"
        sig_files = self.input_folder.list_files(suffix=f"bucket_{bucket:03d}")
        sig_readers = [self.read_sigs(file, file_i) for file_i, file in enumerate(sig_files)]

        pq = [next(sig_reader) for sig_reader in sig_readers]
        heapq.heapify(pq)

        out_f = self.output_folder.open(f"{bucket:05d}.dups", mode="wb")

        last: HashSig | None = None
        while pq:
            v: HashSig = heapq.heappop(pq)
            if last and last.sig == v.sig:
                out_f.write(struct.pack("<4I", last.file_id, last.doc_id, v.file_id, v.doc_id))
            last = v
            next_sig = next(sig_readers[v.file_id], None)
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_buckets = num_buckets

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.input_folder.set_lock(dl_lock)
        self.output_folder.set_lock(up_lock)

    def __call__(self, data: DocumentsPipeline, bucket: int = 0, world_size: int = 1):
        dup_files = self.input_folder.list_files(extension=".dups")
        assert len(dup_files) == self.num_buckets, "There should be exactly one .dups file per bucket"
        union_set = {}

        def parent(x):
            if x not in union_set or union_set[x] == x:
                return x
            union_set[x] = parent(union_set[x])
            return union_set[x]

        for dup_file in dup_files:
            with dup_file.open(binary=True) as df:
                while data := df.read(4 * 4):
                    f1, d1, f2, d2 = struct.unpack("<4I", data)[0]
                    a, b = (f1, d1), (f2, d2)
                    union_set[parent(a)] = parent(b)

        for node, p in sorted(union_set.items()):
            if node != p:
                file, doc = node
                self.output_folder.open(f"{file:06d}.remove", mode="wb").write(struct.pack("<I", doc))
        self.output_folder.close()


class MinhashDedupFilter(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 4"

    def __init__(
        self,
        input_folder: BaseInputDataFolder,
        output_folder: BaseOutputDataFolder,
        num_buckets: int = DEFAULT_NR_BUCKETS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_folder = input_folder
        self.output_folder = output_folder
        self.num_buckets = num_buckets

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.data_folder.set_lock(dl_lock)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        remove_data = self.data_folder.get_files_shard(rank, world_size)
        assert len(remove_data) == 1, f"Must have exactly one .remove file per task. Found {len(remove_data)} files."

        with remove_data[0].open_binary() as f:

            def get_next():
                data = f.read(4)
                if data:
                    return struct.unpack("<I", data)

            next_removal = get_next()
            for idx, doc in enumerate(data):
                self.stat_update(StatHints.total)
                if next_removal == idx:
                    # to remove
                    next_removal = get_next()
                    continue
                yield doc
