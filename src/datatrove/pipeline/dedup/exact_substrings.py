"""
Here we implement exact substrings as suggested by https://arxiv.org/pdf/2107.06499.pdf.
We use suffix array to deduplicate exact substrings above a minimum threshold. We will take the code tom build the
actual suffix array and find duplicates form the GitHub page of "Deduplicating Training Data Makes Language Models
Better".
TLDR
1) DatasetToSequence map 1 file into a giant sequence S. With unique separators at the end of each doc. It also saves
   a second file with the byte offset of where each individual doc begins.
2) MergeSequences all sequences into a big single sequence.

3) ... call deduplicate-text-datasets scripts ...
4)
5)



2) build suffix array A(S)
3) check for duplicates scanning A(S)
4) filter out duplicates
---

"""
import pickle
import struct
from typing import Generator

import numpy as np
import tokenizers

from datatrove.io import BaseInputDataFolder, BaseOutputDataFolder, InputDataFile
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep
from datatrove.pipeline.readers import JsonlReader

from .utils import ExtensionHelperES as EH


def prepare_doc(tokenizer, doc: str, rank: int, doc_id: int):
    tokens = tokenizer.encode(doc).ids
    tokens = np.array(tokens, dtype=np.uint16)
    b_doc = struct.pack("<I", rank) + b"\xff\xff" + struct.pack("<I", doc_id) + tokens.tobytes()
    return b_doc


class DatasetToSequence(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🪞 - exact-substrings stage 1"

    def __init__(self, output_folder=BaseOutputDataFolder, tokenizer_name: str = "gpt2", **kwargs):
        super().__init__(**kwargs)
        self.output_folder = output_folder
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_name)
        self.doc_lens = [0]

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def save_sizes(self, rank: int):
        f_lens = self.output_folder.open(f"{rank:05d}{EH.stage_1_sequence_size}", mode="wb")
        for size in self.doc_lens[1:]:
            f_lens.file_handler.write(struct.pack("<Q", size))
        f_lens.close()

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        f_sequence = self.output_folder.open(f"{rank:05d}{EH.stage_1_sequence}", mode="wb")
        for i, doc in enumerate(data):
            with self.stats.time_manager:
                b_doc = prepare_doc(tokenizer=self.tokenizer, doc=doc.content, rank=rank, doc_id=i)
                self.doc_lens.append(len(b_doc))
                f_sequence.file_handler.write(b_doc)

        assert i < 2**32, "doc ID overflow"

        self.save_sizes(rank)
        f_sequence.close()


class MergeSequences(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🪞 - exact-substrings stage 2"

    def __init__(self, input_folder: BaseInputDataFolder, output_folder: BaseOutputDataFolder, **kwargs):
        super().__init__(**kwargs)
        self.input_folder = input_folder
        self.output_folder = output_folder

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        bytes_per_sequence = [0]
        with self.stats.time_manager:
            assert world_size == 1, f"{world_size=} can't be greater than 1!"
            all_files: list[InputDataFile] = self.input_folder.list_files(extension=EH.stage_1_sequence)
            f_sequence = self.output_folder.open(f"dataset{EH.stage_2_big_sequence}", mode="wb")
            for file in all_files:
                with file.open(binary=True) as f:
                    sequence = f.read()
                    print(bytes_per_sequence[-1] + len(sequence))
                    bytes_per_sequence.append(bytes_per_sequence[-1] + len(sequence))
                    f_sequence.file_handler.write(sequence)
            f_sequence.close()
            f_bytes = self.output_folder.open(f"bytes_offsets{EH.stage_2_bytes_offset}", mode="wb")
            f_bytes.file_handler.write(np.array([bytes_per_sequence], np.uint32).tobytes())
            f_bytes.close()


def sequence_reader(file: InputDataFile, size_file: InputDataFile) -> Generator[list, None, None]:
    def read_bytes(x):
        # 4 bytes for rank + 2 bytes for  b"\xff\xff" + 4 bytes for doc_id
        return np.frombuffer(x[10:], dtype=np.uint16).tolist()

    with size_file.open(binary=True) as f_size:
        with file.open(binary=True) as f:
            while True:
                n_bytes = f_size.read(8)
                n_bytes = struct.unpack("<Q", n_bytes)[0]
                yield f.read(n_bytes)  # read_bytes()


class DedupReader(JsonlReader):
    type = "🫂 - DEDUP"
    name = "🪞 - exact-substrings stage 3"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        sequence_folder: BaseInputDataFolder,
        gzip: bool = True,
        tokenizer_name: str = "gpt2",
        **kwargs,
    ):
        super().__init__(data_folder=data_folder, gzip=gzip, **kwargs)
        self.sequence_folder = sequence_folder
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_name)
        self.bytes_offset = None
        self.bytes_ranges = None
        self.bytes_counter = 0
        self.idx = 0

    def read_bytes_offset(self, rank: int):
        offset_array_file: InputDataFile = self.sequence_folder.list_files(extension=EH.stage_2_bytes_offset)[0]
        with offset_array_file.open(binary=True) as f:
            offset_array = f.read()
        self.bytes_offset = np.frombuffer(offset_array, dtype=np.uint32)[rank]

    def get_all_files(self, rank: int, world_size: int):
        sequence_file = self.sequence_folder.get_files_shard(rank, world_size, extension=EH.stage_1_sequence)
        size_file = self.sequence_folder.get_files_shard(rank, world_size, extension=EH.stage_1_sequence_size)
        byte_range = self.sequence_folder.get_files_shard(rank, world_size, extension=EH.stage_3_bytes_ranges)

        assert all(
            [len(sequence_file) == 1, len(size_file) == 1, len(byte_range) == 1]
        ), "Need to run with n_tasks = n_files"
        sequence_file, size_file, byte_range = sequence_file[0], size_file[0], byte_range[0]

        with byte_range.open(binary=True) as f_range:
            self.bytes_ranges = pickle.load(f_range)

        return sequence_file, size_file

    def get_range(self, bytes_len: int):
        ranges = []
        while self.bytes_counter < self.bytes_ranges[self.idx][1] - self.bytes_offset < self.bytes_counter + bytes_len:
            ranges.append(self.bytes_ranges[self.idx])
            self.idx += 1
        return ranges

    def normalize_range(self, x):
        return x[0] - self.bytes_offset - self.bytes_counter, x[1] - self.bytes_offset - self.bytes_counter

    def remove_duplicate(self, doc, bytes_content):
        duplicates_ranges = self.get_range(len(bytes_content))
        if duplicates_ranges:
            for dup_range in duplicates_ranges:
                a, b = self.normalize_range(dup_range)
                # TODO understand why b - a is not a multiple of 16
                # print(np.frombuffer(bytes_content[a:b], dtype=np.uint16).tolist())
        self.bytes_counter += len(bytes_content)
        return doc

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        self.read_bytes_offset(rank)
        sequence_file, size_file = self.get_all_files(rank, world_size)
        data = self.read_files_shard(self.data_folder.get_files_shard(rank, world_size))

        for doc, doc_content in zip(data, sequence_reader(sequence_file, size_file)):
            yield self.remove_duplicate(doc, doc_content)
