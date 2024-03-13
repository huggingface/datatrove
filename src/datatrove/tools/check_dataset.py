import argparse
import os
import struct
from typing import IO

import numpy as np
from tqdm import tqdm

from datatrove.io import DataFolder, get_datafolder
from datatrove.utils.tokenization import load_tokenizer


parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="path to folder with dataset to check", nargs="?", default=os.getcwd())
parser.add_argument("-t", "--tokenizer", type=str, help="tokenizer to use", default="gpt2")
parser.add_argument("--eos", type=str, help="eos token", default="<|endoftext|>")

"""
    Checks if there is a eos token at the end of each document, matching the .index file.
    Checks if the sizes of the last doc end, the .ds and the .ds.loss file match
"""


def load_doc_ends(file: IO):
    """
        Reads a list of uint64s from a binary file handler
    Args:
      file: IO:

    Returns:

    """
    with file as f:
        return np.frombuffer(f.read(), dtype=np.uint64).tolist()


def load_dataset_bytes(file, doc_ends, bytes_per_value: int = 2):
    """
        Reads tokens directly from a binary file using doc_ends to read one document at a time.
    Args:
      file: file handler
      doc_ends: list with ending positions of each document
      bytes_per_value: int:  (Default value = 2)  how many bytes to read per token

    Returns:

    """
    with file as f:
        for start, end in zip([0] + doc_ends[:-1], doc_ends):
            data = f.read((end - start) * bytes_per_value)
            assert len(data) == (end - start) * bytes_per_value, "Could not read correct number of bytes"
            yield data
        assert f.read(1) == b"", "Dataset should be exhausted but there is more data to read"


def check_dataset(input_folder: DataFolder, tokenizer: str = "gpt2", eos_token: str = "<|endoftext|>"):
    """
    Reads a dataset and checks if loss tokens match up to the corresponding doc ends files
    Args:
      input_folder: DataFolder:
      tokenizer: str:  (Default value = "gpt2")
      eos_token: str:  (Default value = "<|endoftext|>")

    Returns:

    """
    tokenizer = load_tokenizer(tokenizer)

    eos_token = tokenizer.token_to_id(eos_token)

    def open_file(path):
        return input_folder.open(path, "rb")

    datafiles = input_folder.list_files(glob_pattern="*.ds")
    datafiles_index = input_folder.list_files(glob_pattern="*.ds.index")
    datafiles_loss = input_folder.list_files(glob_pattern="*.ds.loss")
    check_loss = not not datafiles_loss
    assert len(datafiles) == len(datafiles_index) and (not check_loss or len(datafiles) == len(datafiles_loss)), (
        "Mismatch between number of .ds, " ".ds.index and/or .ds.loss files"
    )

    doc_ends = [load_doc_ends(open_file(file)) for file in datafiles_index]
    token_inputs = [load_dataset_bytes(open_file(path), ends) for path, ends in zip(datafiles, doc_ends)]
    loss_inputs = (
        [load_dataset_bytes(open_file(path), ends, bytes_per_value=1) for path, ends in zip(datafiles_loss, doc_ends)]
        if check_loss
        else [None] * len(token_inputs)
    )
    for filei, (file_doc_ends, file_token_inputs, file_loss_inputs) in enumerate(
        zip(doc_ends, token_inputs, loss_inputs)
    ):
        for doci, tokens in tqdm(enumerate(file_token_inputs), total=len(file_doc_ends)):
            last_token = struct.unpack("<H", tokens[-2:])[0]
            assert last_token == eos_token, f"no EOS at doc end of doc {doci}"


if __name__ == "__main__":
    args = parser.parse_args()

    input_folder: DataFolder = get_datafolder(args.data)

    check_dataset(input_folder, args.tokenizer, args.eos)
    print("All checks ok")
