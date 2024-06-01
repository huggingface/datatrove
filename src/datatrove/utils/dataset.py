from bisect import bisect

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

        Args:
            file_path (str): path to file on s3, locally, or some other fsspec supported path
            seq_len (int): sequence length
            token_size (int): size of a single token, in bytes. Usually 2 for vocab sizes < 65k and 4 for larger
            max_tokens (int): only read at most this number of tokens
        """

        def __init__(
            self,
            file_path: str,
            seq_len: int,
            token_size: int = 2,
            max_tokens: int | None = None,
        ):
            self.file_path: str = file_path
            self.seq_len = seq_len
            self.token_size = token_size

            self.fs: AbstractFileSystem
            self.fs, self.file_path = url_to_fs(file_path)
            fsize = self.fs.size(self.file_path)
            # total number of full contexts in this file
            num_tokens = fsize // self.token_size
            self._len = (min(max_tokens, num_tokens) if max_tokens else num_tokens) // (seq_len + 1)
            self._f = None

        def __getitem__(self, item):
            if not self._f:
                self._f = self.fs.open(self.file_path, "rb")
            chunk_size = self.token_size * (self.seq_len + 1)
            self._f.seek(item * chunk_size)
            return {
                "input_ids": torch.as_tensor(
                    np.frombuffer(self._f.read(chunk_size), np.uint16 if self.token_size == 2 else np.uint32).astype(
                        np.int64
                    ),
                    dtype=torch.long,
                )
            }

        def __len__(self):
            return self._len

        def __del__(self):
            if self._f:
                self._f.close()

    class DatatroveFolderDataset(Dataset):
        """
        Dataset for a folder of .ds files
        We loop on the dataset if asking for an index larger than the dataset size

        Args:
            folder_path (str): path to folder on S3, locally, or some other fsspec supported path
            seq_len (int): sequence length
            filename_pattern (Union[Pattern, str], optional): filename pattern. Defaults to None.
            recursive (bool, optional): search recursively. Defaults to True.
            token_size (int): size of a single token, in bytes. Usually 2 for vocab sizes < 65k and 4 for larger
            max_tokens (int): only read at most this number of tokens
            shuffle (bool, optional): shuffle the files in the folder. Defaults to False.
            seed (int, optional): seed for shuffling. Defaults to 42.
        """

        def __init__(
            self,
            folder_path: str,
            seq_len: int,
            filename_pattern: str = None,
            recursive: bool = True,
            token_size: int = 2,
            max_tokens: int | None = None,
            shuffle: bool = False,
            seed: int = 42,
        ):
            self.folder_path = folder_path
            self.filename_pattern = filename_pattern
            fs, folder_path = url_to_fs(folder_path)
            matched_files = (
                fs.find(folder_path, detail=False, maxdepth=1 if not recursive else None)
                if not filename_pattern
                else fs.glob(filename_pattern, maxdepth=1 if not recursive else None)
            )
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
else:
    DatatroveFileDataset = NotImplemented
    DatatroveFolderDataset = NotImplemented
