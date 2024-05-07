# https://github.com/ekzhu/datasketch/blob/master/datasketch/hashfunc.py
import hashlib
import struct
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import xxhash

from datatrove.utils._import_utils import guarded_import


def sha1_hash32(data: str):
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    return struct.unpack("<I", hashlib.sha1(data.encode("utf-8")).digest()[:4])[0]


def sha1_hash64(data: str):
    """A 64-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 64-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 64 bits.
    """
    return struct.unpack("<Q", hashlib.sha1(data.encode("utf-8")).digest()[:8])[0]


def xxhash32(data: str):
    return xxhash.xxh32_intdigest(data)  # type: ignore


def xxhash64(data: str):
    return xxhash.xxh64_intdigest(data)  # type: ignore


@dataclass
class HashConfig:
    precision: Literal[32, 64] = 64
    hash_fc: Literal["sha1", "xxhash"] = "xxhash"

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


def create_hash_func(config: HashConfig) -> Callable[[str], int]:
    # TODO: Check requirements for xxhash
    if config.hash_fc == "sha1":
        return sha1_hash32 if config.precision == 32 else sha1_hash64
    elif config.hash_fc == "xxhash":
        guarded_import("Hash Config", "xxhash")
        return xxhash32 if config.precision == 32 else xxhash64
    else:
        raise ValueError(f"Unknown {config.hash_fc=}")


DEFAULT_HASH_CONFIG = HashConfig()
