import argparse
import mmap
import os
import struct

import numpy as np
from tokenizers import Tokenizer

from datatrove.io import BaseInputDataFolder, InputDataFile


parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="path to folder with dataset to check", nargs="?", default=os.getcwd())
parser.add_argument("-t", "--tokenizer", type=str, help="tokenizer to use", default="gpt2")
parser.add_argument("--eos", type=str, help="eos token", default="<|endoftext|>")

"""
    Checks if there is a eos token at the end of each document, matching the .index file.
    Checks if the sizes of the last doc end, the .ds and the .ds.loss file match
"""


def load_doc_ends(file: InputDataFile):
    with file.open_binary() as f:
        return np.frombuffer(f.read(), dtype=np.uint64)


def load_input_mmap(file: InputDataFile):
    with file.open_binary() as f:
        return mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)


if __name__ == "__main__":
    args = parser.parse_args()
    input_folder: BaseInputDataFolder = BaseInputDataFolder.from_path(args.data)
    tokenizer = Tokenizer.from_pretrained(args.tokenizer)

    eos_token = tokenizer.token_to_id(args.eos)

    datafiles = input_folder.list_files(extension=".ds")
    datafiles_index = input_folder.list_files(extension=".ds.index")
    datafiles_loss = input_folder.list_files(extension=".ds.loss")
    check_loss = not not datafiles_loss
    assert len(datafiles) == len(datafiles_index) and (not check_loss or len(datafiles) == len(datafiles_loss)), (
        "Mismatch between number of .ds, " ".ds.index and/or .ds.loss files"
    )

    doc_ends = [load_doc_ends(file) for file in datafiles_index]
    token_inputs = list(map(load_input_mmap, datafiles))
    loss_inputs = list(map(load_input_mmap, datafiles_loss)) if check_loss else None
    for file_doc_ends, file_token_inputs, file_loss_inputs in zip(doc_ends, token_inputs, loss_inputs):
        assert (
            not check_loss or file_token_inputs.size() == file_loss_inputs.size() * 2
        ), "Mismatch between loss and tokens file sizes"
        assert file_token_inputs.size() == file_doc_ends[-1] * 2, "Size of .ds does not match last doc_end"
        for doci, doc_end in enumerate(file_doc_ends):
            last_token = struct.unpack("<H", file_token_inputs[(doc_end.item() - 1) * 2 : doc_end.item() * 2])[0]
            assert last_token == eos_token, f"no EOS at doc end of doc {doci}"
        file_token_inputs.close()
    print("All checks ok")
