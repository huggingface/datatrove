import hashlib
import os
import re
import struct
import unicodedata
from functools import cache
from typing import BinaryIO

from fsspec.spec import AbstractBufferedFile


class ExtensionHelperSD:
    stage_1_signature = ".c4_sig"
    stage_2_duplicates = ".c4_dup"
    index = ".c4_index"


class ExtensionHelperES:
    stage_1_sequence = ".es_sequence"
    stage_1_sequence_size = ".es_sequence.size"
    stage_2_big_sequence = ".big_sequence"
    stage_2_bytes_offset = ".info"
    stage_3_bytes_ranges = ".bytearange"


PUNCTUATION = "!/—”:％１〈&(、━\\【#%「」，】；+^]~“《„';’{|∶´[=-`*．（–？！：$～«〉,><》)?）。…@_.\"}►»" + "".join(
    map(chr, list(range(0, 32)) + list(range(127, 160)))
)


def read_tuples_from_file(file: BinaryIO, *formats, lines_to_buffer: int = 5):
    """Utility to easily parse binary files. formats is a list of struct format characters.
        yields tuples of size len(formats) with the data read

    Args:
        file: the file to read from
        *formats: list of struct format chars. Example, for 2 uint32 and 1 uint64: ['I', 'I', 'Q']
        lines_to_buffer: number of lines to read at a time

    Returns:tuples with data specified in formats

    """
    if lines_to_buffer != -1 and lines_to_buffer < 1:
        raise ValueError("lines_to_buffer must be >= 1 or -1 (for unlimited)")
    fstring = "<" + "".join(formats)
    reader = struct.Struct(fstring)
    while True:
        chunk = file.read(lines_to_buffer * reader.size if lines_to_buffer != -1 else -1)
        if not chunk:
            break
        yield from reader.iter_unpack(chunk)


def simplify_text(text: str) -> str:
    """Performs the following operations to increase recall when looking for matches between documents:
    - lowercase text
    - replace all whitespace with a single " "
    - remove all punctuation
    - convert diacritics
    - unicode normalize

    Args:
        text

    Returns:
        modified text
    """
    # lower case
    text = text.lower()
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    text = re.sub(r"\s+", " ", text.strip())
    # remove punctuation
    text = text.translate(str.maketrans("", "", PUNCTUATION))
    # diacritics/unicode normalization
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text.strip()


# https://github.com/ekzhu/datasketch/blob/master/datasketch/hashfunc.py
def sha1_hash32(data):
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def sha1_hash64(data):
    """A 64-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 64-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 64 bits.
    """
    return struct.unpack("<Q", hashlib.sha1(data).digest()[:8])[0]


def seek_to_start(f: AbstractBufferedFile, start_hash: int, line_format: str, hash_format: str):
    if start_hash == 0:
        return
    line_size = struct.calcsize(line_format)
    nr_lines = f.size // line_size

    @cache
    def read_line_start(line):
        assert 0 <= line < nr_lines
        f.seek(line * line_size, os.SEEK_SET)
        return struct.unpack(hash_format, f.read(struct.calcsize(hash_format)))[0]

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
