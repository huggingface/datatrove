import os
from bisect import bisect
from collections import deque
import random
import fnmatch
from time import sleep

import numpy as np
import torch
from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs

from datatrove.utils._import_utils import is_torch_available


if is_torch_available():
    from torch.utils.data import Dataset

    class DatatroveFileDataset(Dataset):
        """Dataset for a single .ds file created by datatrove
        We loop on the dataset if asking for an index larger than the dataset size
        NOTE that this is heavily optimized for sequential reads, and we actually pre-shuffled the data (assuming it was tokenized with datatrove)
        random access WILL BE SLOW, but should still work

        Args:
            file_path (str): path to file on s3, locally, or some other fsspec supported path
            seq_len (int): sequence length
            token_size (int): size of a single token, in bytes. Usually 2 for vocab sizes < 65k and 4 for larger
            max_tokens (int): only read at most this number of tokens
            read_path (str): path to local file/copy to read from. If it exists, we read from this file instead of from file_path. Useful when we offload some data to remote and only keep the needed files on disk.
        """

        def __init__(
            self,
            file_path: str,
            seq_len: int,
            token_size: int = 2,
            max_tokens: int | None = None,
            return_positions: bool = False,
            eos_token_id: int | None = None,
            read_path: str | None = None,
            fsize: int | None = None,  # Add fsize parameter
        ):
            self.file_path: str = file_path
            self.read_path = read_path
            self.seq_len = seq_len
            self.token_size = token_size
            self.return_positions = return_positions
            self.eos_token_id = eos_token_id
            self.fs: AbstractFileSystem
            self.fs, self.file_path = url_to_fs(file_path)
            
            # Use provided fsize if available, otherwise fetch it
            if fsize is None:
                fsize = self.fs.size(self.file_path)
                
            # total number of full contexts in this file
            num_tokens = fsize // self.token_size
            self._len = (min(max_tokens, num_tokens) if max_tokens else num_tokens) // (seq_len + 1)
            # once we're done getting the metadata (length), we can now rely on read_path
            if read_path:
                self.fs, self.file_path = url_to_fs(read_path)
                
            self._f = None
            self._f_pos = None
            self._idx_buffer = None
            self._last_item = None

        def _get_pos(self, item):
            """
            Reads document ends from .index and returns positions for the entire window.
            For example, if the documents in the window end at token positions [3, 5, 8], for seq_len+1=10,the positions will be [0, 1, 2, 0, 1, 0, 1, 2, 0, 1]
            """
            # Calculate token window range
            window_start = item * (self.seq_len + 1)
            window_end = window_start + self.seq_len  # exclusive, but .index is also exclusive

            # Initialize file if first access
            if self._last_item is None or item < self._last_item:
                if self._f_pos is None:
                    self._f_pos = self.fs.open(self.file_path + ".index", "rb")
                self._idx_buffer = deque()
                # we could binary search but we are assuming sequential reads (which is what we optimized for by pre-shuffling the data), so we always read from the start
                self._f_pos.seek(0)

            # 1. Drop positions before the window
            while self._idx_buffer and self._idx_buffer[0] < window_start:
                self._idx_buffer.popleft()

            # 2. Read until we have at least one position beyond the window or EOF
            while not self._idx_buffer or self._idx_buffer[-1] <= window_end:
                buffer = self._f_pos.read(1024 * 8)  # uint64 = 8 bytes

                if not buffer:
                    break  # End of file

                self._idx_buffer.extend(np.frombuffer(buffer, np.uint64))

            # 3. Extract positions within the window and convert to local indices
            doc_ends = torch.tensor(
                [0] + [pos - window_start for pos in self._idx_buffer if window_start < pos <= window_end],
                dtype=torch.int,
            )

            # get actual positions
            # example: doc_ends = [0, 3, 5, 8]. seq_len+1=10
            # pos = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            # prev_ends = [-1, 0, 3, 5]
            # offsets = [0, -2, -1, -2]
            # pos = [0, 1, 1, -2, 1, -1, 1, 1, -2, 1]
            # cumsum = [0, 1, 2, 0, 1, 0, 1, 2, 0, 1]

            pos = torch.ones(self.seq_len + 1, dtype=torch.int)
            prev_ends = torch.cat([torch.tensor([-1], dtype=torch.int), doc_ends[:-1]])
            offsets = prev_ends - doc_ends + 1
            pos[doc_ends] = offsets
            assert pos[0] == 0, "First position should be 0"
            return torch.cumsum(pos, dim=0)

        def _get_positions_from_tokens(self, tokens):
            pos = torch.ones_like(tokens, dtype=torch.int64)
            doc_ends = torch.cat(
                [torch.tensor([0], dtype=torch.int64), torch.where(tokens[:-1] == self.eos_token_id)[0] + 1]
            )
            prev_ends = torch.cat([torch.tensor([-1], dtype=torch.int64), doc_ends[:-1]])
            offsets = prev_ends - doc_ends + 1
            pos[doc_ends] = offsets
            assert pos[0] == 0, "First position should be 0"
            return torch.cumsum(pos, dim=0)

        def _get_input_ids(self, item):
            if self._f is None:
                self._f = self.fs.open(self.file_path, "rb")
            chunk_size = self.token_size * (self.seq_len + 1)
            self._f.seek(item * chunk_size)
            return torch.as_tensor(
                np.frombuffer(self._f.read(chunk_size), np.uint16 if self.token_size == 2 else np.uint32).astype(
                    np.int64
                ),
                dtype=torch.long,
            )

        def __getitem__(self, item):
            data = {"input_ids": self._get_input_ids(item)}
            if self.return_positions:
                data["positions"] = (
                    self._get_pos(item)
                    if self.eos_token_id is None
                    else self._get_positions_from_tokens(data["input_ids"])
                )
            self._last_item = item
            return data

        def __len__(self):
            return self._len

        def __del__(self):
            if self._f:
                self._f.close()
            if self._f_pos:
                self._f_pos.close()

    class DatatroveFolderDataset(Dataset):
        """
        Dataset for a folder of .ds files
        We loop on the dataset if asking for an index larger than the dataset size

        Args:
            folder_path (str): path to folder on S3, locally, or some other fsspec supported path
            seq_len (int): sequence length
            filename_pattern (Union[Pattern, str], optional): filename pattern. Defaults to **/*.ds.
            recursive (bool, optional): search recursively. Defaults to True.
            token_size (int): size of a single token, in bytes. Usually 2 for vocab sizes < 65k and 4 for larger
            max_tokens (int): only read at most this number of tokens
            shuffle (bool, optional): shuffle the files in the folder. Defaults to False.
            seed (int, optional): seed for shuffling. Defaults to 42.
            folder_read_path (str): path to local file/copy to read from. If it exists, we read from this folder instead of from folder_path. Useful when we offload some data to remote and only keep the needed files on disk.
        """

        def __init__(
            self,
            folder_path: str,
            seq_len: int,
            filename_pattern: str = "**/*.ds",
            recursive: bool = True,
            token_size: int = 2,
            max_tokens: int | None = None,
            shuffle: bool = False,
            seed: int = 42,
            return_positions: bool = False,
            eos_token_id: int | None = None,
            read_path: str | None = None,
            matched_files: list | None = None,  # Add matched_files parameter
            file_sizes: dict | None = None,  # Add file_sizes parameter
        ):
            self.folder_path = folder_path
            self.filename_pattern = filename_pattern
            self.return_positions = return_positions
            fs, folder_path = url_to_fs(folder_path)

            if matched_files is None:
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        matched_files = (
                            fs.find(folder_path, detail=False, maxdepth=1 if not recursive else None)
                            if not filename_pattern
                            else fs.glob(os.path.join(folder_path, filename_pattern), maxdepth=1 if not recursive else None)
                        )
                        matched_files = sorted(matched_files)
                        break
                    except OSError as e:
                        if "Please reduce your request rate" in str(e) and attempt < max_retries - 1:
                            sleep_time = (2 ** attempt) + random.random()
                            sleep(sleep_time)
                            continue
                        raise

            if not matched_files:
                raise FileNotFoundError(f'No files matching "{filename_pattern}" found in {folder_path}')

            self.files = []
            remaining_tokens = max_tokens
            for path in matched_files:
                file_data = DatatroveFileDataset(
                    fs.unstrip_protocol(path),
                    seq_len,
                    token_size=token_size,
                    max_tokens=remaining_tokens,
                    return_positions=return_positions,
                    eos_token_id=eos_token_id,
                    read_path=os.path.join(read_path, os.path.relpath(path, folder_path)) if read_path else None,
                    fsize=file_sizes.get(path) if file_sizes else None,
                )
                self.files.append(file_data)
                if remaining_tokens is not None:
                    remaining_tokens -= len(file_data) * (seq_len + 1)
                    if remaining_tokens <= 0:
                        break

            if shuffle:
                rand = np.random.default_rng(seed)
                ordering = rand.permutation(range(len(self.files)))
                self.files = [self.files[i] for i in ordering]

            self.lens = np.cumsum([0] + [len(f) for f in self.files]).tolist()

            self.current_file = 0

        def __getitem__(self, item):
            # check if we are in the same file as before
            if not (self.lens[self.current_file] <= item < self.lens[self.current_file + 1]):
                # figure out current file
                self.current_file = bisect(self.lens, item) - 1
            # subtract file starting offset
            return self.files[self.current_file][item - self.lens[self.current_file]]

        def __len__(self):
            return self.lens[-1] if self.lens else 0
        
        @property
        def current_file_path(self):
            return self.files[self.current_file].read_path if self.files[self.current_file].read_path is not None else self.files[self.current_file].file_path
else:
    DatatroveFileDataset = NotImplemented
    DatatroveFolderDataset = NotImplemented
