import math

import numpy as np
from nltk import ngrams, word_tokenize

from datatrove.data import DocumentsPipeline
from datatrove.io import BaseOutputDataFolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup.utils import sha1_hash32, simplify_content
from datatrove.utils.typeshelper import StatHints


# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = np.uint64((1 << 61) - 1)
_max_hash = np.uint64((1 << 32) - 1)
_hash_range = 1 << 32


def get_optimal_k(size_in_bytes: int, expected_elements: int) -> int:
    m = size_in_bytes * 8
    k = (m / expected_elements) * np.ln(2)
    return math.ceil(k)


def get_false_positive_prob(size_in_bytes: int, n: int, k: int) -> float:
    m = size_in_bytes * 8
    return (1.0 - (1.0 - (1.0 / m)) ** (k * n)) ** k


def suggest_size_in_bytes(expected_elements: int, target_prob=0.01) -> int:
    size_in_bytes = 1024 * 1024
    while (
        get_false_positive_prob(size_in_bytes, expected_elements, get_optimal_k(size_in_bytes, expected_elements))
        > target_prob
    ):
        size_in_bytes *= 2
    return size_in_bytes


class BloomFilterCreation(PipelineStep):
    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 1"

    def __init__(
        self, output_folder: BaseOutputDataFolder, m: int, min_n_grams: int = 5, max_n_grams: int = 13, **kwargs
    ):
        """

        :param output_folder: folder where signatures are saved
        :param n_sentences: n_sentences where duplicates are checked.
        :param stage_2_workers: TODO implement parallel second stage
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_folder = output_folder
        self.m = m
        self.min_n_grams = min_n_grams
        self.max_n_grams = max_n_grams
        self.k = get_optimal_k(
            self.m,
        )

    @property
    def parameters(self):
        if not self._parameters:
            # Create parameters for a random bijective permutation function
            # that maps a 32-bit hash value to another 32-bit hash value.
            # http://en.wikipedia.org/wiki/Universal_hashing
            gen = np.random.RandomState(self.seed)
            self._parameters = gen.randint(1, _mersenne_prime, dtype=np.uint64, size=(1, self.k)), gen.randint(
                0, _mersenne_prime, dtype=np.uint64, size=(1, self.k)
            )
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
        self.signatures = []
        for doc_idx, doc in enumerate(data):
            with self.stats.time_manager:
                self.stat_update(StatHints.total)
                # todo my shit
        self.save_hashes(rank)
        self.output_folder.close()
