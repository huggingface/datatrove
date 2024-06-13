import contextlib
import math
from dataclasses import dataclass, field

import numpy as np

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.hashing import HashConfig, create_hash_func
from datatrove.utils.logging import logger
from datatrove.utils.text import TextNormConfig, ngrams, simplify_text
from datatrove.utils.typeshelper import Languages, StatHints
from datatrove.utils.word_tokenizers import load_word_tokenizer


# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = np.uint64((1 << 61) - 1)
MAX_HASH = 1 << 32 - 1


@dataclass
class BloomFilterConfig:
    """
    m_bytes: bloom filter size in bytes (actual size x8 bigger)
    k: number of hashes
    expected_elements: expected number of elements, aka
        shingles.
    duplicate_threshold: above which documents are considered as
        duplicated
    n_grams: n_grams to use
    seed: seed
    """

    m_bytes: int
    k: int = None
    expected_elements: int = None
    duplicate_threshold: float = 0.8
    n_grams: int = 13
    seed: int = 0
    norm_config: TextNormConfig = field(default_factory=TextNormConfig)
    hash_config: HashConfig = field(default_factory=lambda: HashConfig(precision=32))

    @property
    def m(self):  # (self.m + 7) // 8  # size in bytes
        return self.m_bytes * 8

    def __post_init__(self):
        if self.k is None:
            self.k = get_optimal_k(self.m, expected_elements=self.expected_elements)


def get_optimal_k(size_in_bytes: int, expected_elements: int) -> int:
    assert expected_elements, f"if {expected_elements=} then k must be given"
    m = size_in_bytes * 8
    k = (m / expected_elements) * np.log(2)
    return math.ceil(k)


def get_false_positive_prob(size_in_bytes: int, n: int, k: int) -> float:
    m = size_in_bytes * 8
    return (1.0 - (1.0 - (1.0 / m)) ** (k * n)) ** k


class SingleBloomFilter(PipelineStep):
    """Single Bloom filter for deduplication

    Args:
        output_folder: output folder: local or on S3
        save_bloom_filter: if true saves bloom filter for later use
        exclusion_writer: saves duplicated data
    """

    type = "🫂 - DEDUPS"
    name = "🪷 Bloom-filter"

    def __init__(
        self,
        output_folder: DataFolderLike,
        config: BloomFilterConfig,
        save_bloom_filter: bool = False,
        exclusion_writer: DiskWriter = None,
        language: str = Languages.english,
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.tokenizer = load_word_tokenizer(language)
        self.config = config
        self.bit_vector = bytearray(([0] * self.config.m_bytes))
        self.save_bloom_filter = save_bloom_filter
        self.exclusion_writer = exclusion_writer
        # TODO: Add support for 64-bit
        assert self.config.hash_config.precision == 32, "Bloom filter only supports 32-bit hashes"
        self.hash_fc = create_hash_func(self.config.hash_config)
        assert self.config.m < MAX_HASH

        self.total_shingles = 0
        self._parameters = None

        assert self.config.m_bytes < MAX_HASH, f"{MAX_HASH=} is smaller than {self.config.m_bytes=}"
        if self.config.expected_elements:
            fp = get_false_positive_prob(self.config.m_bytes, n=self.config.expected_elements, k=self.config.k)
            if fp > 0.05:
                logger.warning(f"False probability = {fp:.3}")
            else:
                logger.info(f"False probability = {fp:.3}")
        self.language = language

    @property
    def parameters(self):
        """Returns the parameters for the hash functions.
            Create parameters for a random bijective permutation function
            that maps a 32-bit hash value to another 32-bit hash value.
            http://en.wikipedia.org/wiki/Universal_hashing

        Returns:
            tuple: (a, b) parameters for the hash functions
                where a and b are numpy uint64 arrays of shape (1, k) containing the
                random parameters for the hash functions.
        """
        if not self._parameters:
            gen = np.random.RandomState(self.config.seed)
            self._parameters = (
                gen.randint(1, _mersenne_prime, dtype=np.uint64, size=(1, self.config.k)),
                gen.randint(0, _mersenne_prime, dtype=np.uint64, size=(1, self.config.k)),
            )
        return self._parameters

    def get_shingles(self, text: str) -> np.ndarray:
        """Get shingles from a string of text
        Shingles are created by hashing n-grams of simplified text (lower cases, whitespace normalized, no punctuation, etc).
        """
        return np.fromiter(
            [
                self.hash_fc(" ".join(x))
                for x in ngrams(
                    self.tokenizer.word_tokenize(simplify_text(text, self.config.norm_config)), self.config.n_grams
                )
            ],
            dtype=np.uint64,
        ).reshape((-1, 1))

    def get_indexes(self, shingles: np.ndarray) -> list[list[int]]:
        """Get indexes for the shingles with the k hashing functions"""
        a, b = self.parameters
        phv = np.bitwise_and((shingles * a + b) % _mersenne_prime, self.config.m_bytes)
        return phv.tolist()

    def update_bf(self, indexes: list[int]):
        """Update the bloom filter with the indexes"""
        for index in indexes:
            byte_index, bit_index = divmod(index, 8)
            mask = 1 << bit_index
            self.bit_vector[byte_index] |= mask

    def query(self, indexes: list[int]) -> bool:
        """Query the bloom filter with the indexes"""
        for idx in indexes:
            byte_index, bit_index = divmod(idx, 8)
            mask = 1 << bit_index
            if (self.bit_vector[byte_index] & mask) == 0:
                return False
        return True

    def step(self, doc: Document) -> bool:
        """Deduplication step
        Compute shingles, indexes, and query the bloom filter
        """
        shingles = self.get_shingles(doc.text)
        self.total_shingles += shingles.size
        if shingles.size == 0:
            return True
        shingle_indexes = self.get_indexes(shingles)

        duplicate_shingles = 0
        indexes_to_update = []
        for indexes in shingle_indexes:
            if self.query(indexes):
                duplicate_shingles += 1
            else:
                indexes_to_update.extend(indexes)

        self.update_bf(indexes_to_update)
        if duplicate_shingles / len(shingles) > self.config.duplicate_threshold:
            self.stat_update(StatHints.dropped)
            return False
        return True

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
            for doc_idx, doc in enumerate(data):
                with self.track_time():
                    self.stat_update(StatHints.total)
                    if not self.step(doc):
                        self.stat_update(StatHints.dropped)
                        if self.exclusion_writer:
                            writer.write(doc, rank)
                        continue
                self.stat_update(StatHints.forwarded)
                yield doc
            if self.save_bloom_filter:
                with self.output_folder.open("bloom_filter.bloom", mode="wb") as f:
                    f.write(self.bit_vector)

        logger.info(f"{self.total_shingles=}")
        logger.info(
            f"False probability = {get_false_positive_prob(self.config.m_bytes, n=self.total_shingles, k=self.config.k):.3}"
        )
        logger.info(f"Optimal K given total shingles = {get_optimal_k(self.config.m_bytes, self.total_shingles)}")
