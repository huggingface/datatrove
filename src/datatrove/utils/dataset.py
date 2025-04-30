import json
from bisect import bisect
from collections import deque

import numpy as np
import torch
from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs

from datatrove.io import file_exists, get_datafolder, open_file
from datatrove.utils._import_utils import is_torch_available


if is_torch_available():
    from torch.utils.data import Dataset

    class DatatroveFileDataset(Dataset):
        """Dataset for a single .ds file created by datatrove

        Args:
            file_path (str): path to file on s3, locally, or some other fsspec supported path
            seq_len (int): sequence length
            token_size (int): size of a single token, in bytes. Usually 2 for vocab sizes < 65k and 4 for larger
            max_tokens (int): only read at most this number of tokens
            return_positions (bool): whether to return positions. Defaults to False.
            positions_from_eos_token_id (int, optional): Token ID to use for calculating positions. If set,
                positions are calculated based on this token ID marking the end of sequences. Defaults to None,
                in which case positions are read from the .index file.
            fsize (int, optional): The file size. If None, it will be fetched using fsspec. Defaults to None.

        We loop on the dataset if asking for an index larger than the dataset size
        NOTE that this is heavily optimized for sequential reads, and we actually pre-shuffled the data (assuming it was tokenized with datatrove)
        random access WILL BE SLOW, but should still work
        """

        def __init__(
            self,
            file_path: str,
            seq_len: int,
            token_size: int = 2,
            max_tokens: int | None = None,
            return_positions: bool = False,
            positions_from_eos_token_id: int | None = None,
            fsize: int | None = None,
        ):
            self.file_path: str = file_path
            self.seq_len = seq_len
            self.token_size = token_size
            self.return_positions = return_positions
            self.positions_from_eos_token_id = positions_from_eos_token_id
            self.fs: AbstractFileSystem
            self.fs, self.file_path = url_to_fs(file_path)
            self.fsize = fsize
            if self.fsize is None:
                self.fsize = self.fs.size(self.file_path)
            # total number of full contexts in this file
            num_tokens = self.fsize // self.token_size
            self._len = (min(max_tokens, num_tokens) if max_tokens else num_tokens) // (seq_len + 1)

            self._f = None
            self._f_pos = None
            self._idx_buffer = None
            self._last_item = None

        def _get_pos_from_index_file(self, item):
            """
            Reads document ends from .index file and returns positions for the requested window.

            Positions represent the index of the token within its document.
            For example, if the documents in the window end at token positions [3, 5, 8], for seq_len+1=10,
            the positions will be [0, 1, 2, 0, 1, 0, 1, 2, 0, 1].

            Args:
                item (int): The index of the window to retrieve positions for.

            Returns:
                torch.Tensor: A tensor containing the positions for the tokens in the window.
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
            return torch.cumsum(pos, dim=0)

        def _get_positions_from_tokens(self, tokens):
            """
            Calculate token positions based on an end-of-sequence token ID.

            Positions reset to 0 after each occurrence of `positions_from_eos_token_id`.

            Args:
                tokens (torch.Tensor): The input token IDs.

            Returns:
                torch.Tensor: A tensor containing the calculated positions.
            """
            pos = torch.ones_like(tokens, dtype=torch.int64)
            doc_ends = torch.cat(
                [
                    torch.tensor([0], dtype=torch.int64),
                    torch.where(tokens[:-1] == self.positions_from_eos_token_id)[0] + 1,
                ]
            )
            prev_ends = torch.cat([torch.tensor([-1], dtype=torch.int64), doc_ends[:-1]])
            offsets = prev_ends - doc_ends + 1
            pos[doc_ends] = offsets
            return torch.cumsum(pos, dim=0)

        def _get_input_ids(self, item):
            """
            Reads and returns the input IDs for the requested item (window).

            Opens the main data file if not already open, seeks to the correct
            position, reads the required chunk of data, and converts it to a tensor.

            Args:
                item (int): The index of the window to retrieve input IDs for.

            Returns:
                torch.Tensor: A tensor containing the input IDs for the window.
            """
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
            """
            Retrieves a data sample (input IDs and optionally positions) for the given index.

            Args:
                item (int): The index of the sample to retrieve.

            Returns:
                dict: A dictionary containing 'input_ids' and optionally 'positions'.
            """
            data = {"input_ids": self._get_input_ids(item)}
            if self.return_positions:
                data["positions"] = (
                    self._get_pos_from_index_file(item)
                    if self.positions_from_eos_token_id is None
                    else self._get_positions_from_tokens(data["input_ids"])
                )
                assert data["positions"][0] == 0, "First position should be 0"
            self._last_item = item
            return data

        def __len__(self):
            """
            Returns the total number of full sequences in the dataset.
            """
            return self._len

        def __del__(self):
            """
            Closes the open file handles upon object deletion.
            """
            if self._f:
                self._f.close()
            if self._f_pos:
                self._f_pos.close()

    class DatatroveFolderDataset(Dataset):
        """
        Dataset for a folder of .ds files created by datatrove.

        Args:
            folder_path (str): path to folder on S3, locally, or some other fsspec supported path
            seq_len (int): sequence length
            filename_pattern (str, optional): Glob pattern to match files within the folder. Defaults to ".ds".
            recursive (bool, optional): Search recursively within the folder. Defaults to True.
            token_size (int): size of a single token, in bytes. Usually 2 for vocab sizes < 65k and 4 for larger.
            max_tokens (int, optional): Only read at most this number of tokens across all files. Defaults to None.
            shuffle (bool, optional): Shuffle the order of files found in the folder. Defaults to False.
            seed (int, optional): Seed for shuffling. Defaults to 42.
            return_positions (bool, optional): Whether to return positions. Defaults to False.
            positions_from_eos_token_id (int, optional): Token ID to use for calculating positions. If set,
                positions are calculated based on this token ID marking the end of sequences. Defaults to None,
                in which case positions are read from the .index file.
            paths_file (str, optional): Path to a JSON file containing a list of file paths and their sizes.
                If provided and exists, it's used instead of listing files from the filesystem.
                If provided but doesn't exist, it will be created after listing files. Defaults to None.

        We loop on the dataset if asking for an index larger than the dataset size.
        """

        def __init__(
            self,
            folder_path: str,
            seq_len: int,
            filename_pattern: str = ".ds",
            recursive: bool = True,
            token_size: int = 2,
            max_tokens: int | None = None,
            shuffle: bool = False,
            seed: int = 42,
            return_positions: bool = False,
            positions_from_eos_token_id: int | None = None,
            paths_file: str | None = None,
        ):
            self.folder_path = folder_path
            self.folder_df = get_datafolder(folder_path)
            self.filename_pattern = filename_pattern
            self.return_positions = return_positions
            # load list of paths from paths_file or from the fs
            fsizes = {}
            if not paths_file or not file_exists(paths_file):
                matched_files = self.folder_df.list_files(glob_pattern=filename_pattern, recursive=recursive)
                if not matched_files:
                    raise FileNotFoundError(f'No files matching "{filename_pattern}" found in {folder_path}')
            else:
                with open_file(paths_file, "r") as f:
                    file_data = json.load(f)
                    matched_files = [f["path"] for f in file_data]
                    fsizes = {f["path"]: f["size"] for f in file_data}

            self.files = []
            remaining_tokens = max_tokens
            for path in matched_files:
                file_data = DatatroveFileDataset(
                    self.folder_df.resolve_paths(path),
                    seq_len,
                    token_size=token_size,
                    max_tokens=remaining_tokens,
                    return_positions=return_positions,
                    positions_from_eos_token_id=positions_from_eos_token_id,
                    fsize=fsizes.get(
                        path, None
                    ),  # potentially use a cached size to avoid excessive remote calls/possibly offloaded file
                )
                self.files.append(file_data)
                if remaining_tokens is not None:
                    remaining_tokens -= len(file_data) * (seq_len + 1)
                    if remaining_tokens <= 0:
                        break

            if paths_file and not file_exists(paths_file):
                with open_file(paths_file, "wt") as f:
                    json.dump(
                        [{"path": rel_path, "size": f.fsize} for rel_path, f in zip(matched_files, self.files)], f
                    )

            if shuffle:
                rand = np.random.default_rng(seed)
                ordering = rand.permutation(range(len(self.files)))
                self.files = [self.files[i] for i in ordering]

            self.lens = np.cumsum([0] + [len(f) for f in self.files]).tolist()

            self.current_file = 0

        def __getitem__(self, item):
            """
            Retrieves a data sample from the appropriate file based on the global index.

            Finds the correct file containing the item and returns the sample from that file's dataset.

            Args:
                item (int): The global index of the sample to retrieve.

            Returns:
                dict: A dictionary containing 'input_ids' and optionally 'positions'.
            """
            # check if we are in the same file as before
            if not (self.lens[self.current_file] <= item < self.lens[self.current_file + 1]):
                # figure out current file
                self.current_file = bisect(self.lens, item) - 1
            # subtract file starting offset
            return self.files[self.current_file][item - self.lens[self.current_file]]

        def __len__(self):
            """
            Returns the total number of samples across all files in the dataset.
            """
            return self.lens[-1] if self.lens else 0

else:
    DatatroveFileDataset = NotImplemented
    DatatroveFolderDataset = NotImplemented
