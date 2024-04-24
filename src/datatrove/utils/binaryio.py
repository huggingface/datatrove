import os
import struct
from functools import cache
from typing import BinaryIO

import numpy as np
from fsspec.spec import AbstractBufferedFile


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


def read_np_from_file(
    file: BinaryIO,
    dtype: np.dtype,
    is_local_file: bool = False,
) -> np.ndarray:
    """
    Utility which reads data from a file and returns a numpy array.
    Args:
        file: the file to read from
        dtype: expected dtype of data
        is_local_file: whether the file is a local file (enables optimizations)
    Returns:
        numpy array of data from the file
    """
    with file:
        if is_local_file:
            return np.fromfile(file, dtype=dtype)
        else:
            return np.frombuffer(file.read(), dtype=dtype)


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
