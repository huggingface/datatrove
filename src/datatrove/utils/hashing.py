# https://github.com/ekzhu/datasketch/blob/master/datasketch/hashfunc.py
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.hashes.sha1 import sha1_hash32, sha1_hash64


@dataclass(frozen=True)
class HashConfig:
    precision: Literal[32, 64] = 64
    hash_fc: Literal["sha1", "xxhash"] = "xxhash"

    def __post_init__(self):
        if self.hash_fc == "xxhash":
            check_required_dependencies("xxhash Hashing", ["xxhash"])

    @property
    def np_dtype(self):
        return np.uint32 if self.precision == 32 else np.uint64

    @property
    def np_descr(self):
        return np.dtype(self.np_dtype).descr[0][1]

    @property
    def struct_format(self):
        return "I" if self.precision == 32 else "Q"

    @property
    def max(self):
        return np.iinfo(self.np_dtype).max

    @property
    def min(self):
        return np.iinfo(self.np_dtype).min

    def __str__(self):
        return f"HashConfig(precision={self.precision}, hash_fc={self.hash_fc})"


def create_hash_func(config: HashConfig) -> Callable[[str], int]:
    if config.hash_fc == "sha1":
        return sha1_hash32 if config.precision == 32 else sha1_hash64
    elif config.hash_fc == "xxhash":
        from datatrove.utils.hashes.xxhash import xxhash32, xxhash64

        return xxhash32 if config.precision == 32 else xxhash64
    else:
        raise ValueError(f"Unknown {config.hash_fc=}")
