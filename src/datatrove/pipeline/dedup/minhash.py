import hashlib
import struct

import numpy as np

from datatrove.data import DocumentsPipeline
from datatrove.io import BaseOutputDataFolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = np.uint64((1 << 61) - 1)
_max_hash = np.uint64((1 << 32) - 1)
_hash_range = 1 << 32


def sha1_hash32(data):
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def _init_permutations(num_perm):
    # Create parameters for a random bijective permutation function
    # that maps a 32-bit hash value to another 32-bit hash value.
    # http://en.wikipedia.org/wiki/Universal_hashing
    gen = np.random.RandomState(1)
    return gen.randint(1, _mersenne_prime, dtype=np.uint64, size=(1, num_perm)), gen.randint(
        0, _mersenne_prime, dtype=np.uint64, size=(1, num_perm)
    )


class MinhashDedupSignature(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 1"

    def __init__(
        self, output_folder: BaseOutputDataFolder, num_hashes: int = 20, n_grams: int = 5, seed: int = 1, **kwargs
    ):
        super().__init__(**kwargs)
        self.output_folder = output_folder
        self.n_grams = n_grams
        self.num_hashes = num_hashes
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
        return np.min(phv, axis=0).astype(np.uint32)

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc_idx, doc in enumerate(data):
            self.stat_update(StatHints.total)
            # shingles = [" ".join(x) for x in ngrams(word_tokenize(simplify_content(doc.content)), self.n_grams)]
            # self.get_signature(shingles)

        self.output_folder.close()
