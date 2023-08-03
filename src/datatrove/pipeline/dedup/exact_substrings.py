"""
Here we implement exact substrings as suggested by https://arxiv.org/pdf/2107.06499.pdf.
We use suffix array to deduplicate exact substrings above a minimum threshold. We will take the code tom build the
actual suffix array and find duplicates form the GitHub page of "Deduplicating Training Data Makes Language Models
Better".
TLDR
1) DatasetToSequence map 1 file into a sequence S. With unique separators at the beginning of each doc. It also saves
   a second file with the bytes offset of where each individual doc begins.
2) MergeSequences all sequences into a big single sequence. it saves the bytes offset per file.

 ... call deduplicate-text-datasets scripts ...

3) DedupReader reads docs and ranges at the same time and remove duplicates.


---

"""
import struct
from typing import Generator

import numpy as np
import tokenizers
from loguru import logger
from nltk.tokenize import word_tokenize

from datatrove.io import BaseInputDataFolder, BaseOutputDataFolder, InputDataFile
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep
from datatrove.pipeline.readers import JsonlReader
from datatrove.utils.utils import get_language

from .utils import ExtensionHelperES as EH


SEPARATOR_BYTES = 12


def prepare_doc(tokenizer, doc: str, rank: int, doc_id: int):
    tokens = tokenizer.encode(doc).ids
    tokens = np.array(tokens, dtype=np.uint16)
    b_doc = b"\xff\xff" + struct.pack("<I", doc_id) + b"\xff\xff" + struct.pack("<I", rank) + tokens.tobytes()
    return b_doc


class DatasetToSequence(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🪞 - exact-substrings stage 1"

    def __init__(self, output_folder=BaseOutputDataFolder, tokenizer_name: str = "gpt2", **kwargs):
        super().__init__(**kwargs)
        self.output_folder = output_folder
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_name)

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def save_sizes(self, doc_lens: list[int], rank: int):
        f_lens = self.output_folder.open(f"{rank:05d}{EH.stage_1_sequence_size}", mode="wb")
        f_lens.file_handler.write(struct.pack("Q" * len(doc_lens), *doc_lens))

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        # TODO talk with guilherme
        doc_lens = []
        f_sequence = self.output_folder.open(f"{rank:05d}{EH.stage_1_sequence}", mode="wb")
        for i, doc in enumerate(data):
            with self.stats.time_manager:
                b_doc = prepare_doc(tokenizer=self.tokenizer, doc=doc.content, rank=rank, doc_id=i)
                doc_lens.append(len(b_doc))
                f_sequence.file_handler.write(b_doc)

        assert i < 2**32, "doc ID overflow"  # TODO check
        assert i + 1 == len(doc_lens), f"{i=} but {len(doc_lens)=}"

        self.save_sizes(doc_lens, rank)
        self.output_folder.close()


class MergeSequences(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🪞 - exact-substrings stage 2"

    def __init__(
        self,
        input_folder: BaseInputDataFolder,
        output_folder: BaseOutputDataFolder,
        tasks_stage_1: int,
        bytes_per_batch: int = int(500e6),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.tasks_stage_1 = tasks_stage_1
        self.bytes_per_batch = bytes_per_batch

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        bytes_per_sequence = [0]
        with self.stats.time_manager:
            assert world_size == 1, f"{world_size=} can't be greater than 1!"
            all_files: list[InputDataFile] = self.input_folder.list_files(extension=EH.stage_1_sequence)
            assert len(all_files) == self.tasks_stage_1
            f_sequence = self.output_folder.open(f"dataset{EH.stage_2_big_sequence}", mode="wb")
            for file in all_files:
                len_sequence = 0
                with file.open(binary=True) as f:
                    while True:
                        sequence = f.read(self.bytes_per_batch)
                        f_sequence.file_handler.write(sequence)
                        len_sequence += len(sequence)
                        if len(sequence) != self.bytes_per_batch:
                            break
                    bytes_per_sequence.append(bytes_per_sequence[-1] + len_sequence)

            f_bytes = self.output_folder.open(f"bytes_offsets{EH.stage_2_bytes_offset}", mode="wb")
            f_bytes.file_handler.write(np.array([bytes_per_sequence], np.uint32).tobytes())
            self.output_folder.close()


def read_bytes(x):
    # 4 bytes for rank + 4 bytes for  2 * b"\xff\xff" + 4 bytes for doc_id
    return np.frombuffer(x[SEPARATOR_BYTES:], dtype=np.uint16).tolist()


def sequence_reader(file: InputDataFile, size_file: InputDataFile) -> Generator[list, None, None]:
    with size_file.open(binary=True) as f_size:
        with file.open(binary=True) as f:
            while True:
                n_bytes = f_size.read(struct.calcsize("<Q"))
                if len(n_bytes) == 0:
                    break
                assert len(n_bytes) == 8
                n_bytes = struct.unpack("<Q", n_bytes)[0]
                yield f.read(n_bytes)


class DedupReader(JsonlReader):
    type = "🫂 - DEDUP"
    name = "🪞 - exact-substrings stage 3"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        sequence_folder: BaseInputDataFolder,
        gzip: bool = True,
        tokenizer_name: str = "gpt2",
        min_doc_words: int = 50,
        **kwargs,
    ):
        super().__init__(data_folder=data_folder, gzip=gzip, **kwargs)
        self.sequence_folder = sequence_folder
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_name)
        self.min_doc_words = min_doc_words
        self.bytes_offset = None
        self.bytes_ranges = None
        self.rank = None
        self.exhausted_ranges = False
        self.bytes_counter = 0
        self.idx = 0

    def reset(self):
        self.bytes_counter = 0
        self.idx = 0
        self.exhausted_ranges = False
        self.bytes_offset = None
        self.bytes_ranges = None
        self.rank = None

    def read_bytes_offset(self):
        offset_array_file: InputDataFile = self.sequence_folder.list_files(extension=EH.stage_2_bytes_offset)[0]
        with offset_array_file.open(binary=True) as f:
            offset_array = f.read()
        self.bytes_offset = np.frombuffer(offset_array, dtype=np.uint32)
        logger.info(f"{self.rank=}, -> {self.bytes_offset[self.rank]=}")

    def read_bytearange(self, bytes_range_file: InputDataFile):
        with bytes_range_file.open(binary=False) as f:
            bytes_ranges = f.read()

        bytes_ranges = bytes_ranges.split("\n")
        for i, x in enumerate(bytes_ranges):
            if x == "out":
                break

        # remove lines until out and remove last empty value
        bytes_ranges = bytes_ranges[i + 1 : -1]

        shard_bytes_ranges = []
        for br in bytes_ranges:
            a, b = br.split(" ")
            a, b = int(a), int(b)
            if b > self.bytes_offset[self.rank + 1] + SEPARATOR_BYTES:
                break
            if b > self.bytes_offset[self.rank]:
                shard_bytes_ranges.append((a, b))
        self.bytes_ranges = shard_bytes_ranges

    # rank is given like that for testing purposes.
    def get_all_files(self, rank: int, world_size: int):
        sequence_file = self.sequence_folder.get_files_shard(rank, world_size, extension=EH.stage_1_sequence)
        size_file = self.sequence_folder.get_files_shard(rank, world_size, extension=EH.stage_1_sequence_size)
        byte_range_file = self.sequence_folder.list_files(extension=EH.stage_3_bytes_ranges)

        assert all(
            [len(sequence_file) == 1, len(size_file) == 1, len(byte_range_file) == 1]
        ), f"Need to run with n_tasks = n_files. {len(sequence_file)=}, {len(sequence_file)=}, {len(byte_range_file)=}"
        sequence_file, size_file, byte_range_file = sequence_file[0], size_file[0], byte_range_file[0]

        self.read_bytearange(byte_range_file)
        return sequence_file, size_file

    def get_range(self, bytes_len: int):
        ranges = []
        lim = self.bytes_counter + bytes_len
        if self.exhausted_ranges:
            return ranges

        while (
            self.bytes_counter < self.bytes_ranges[self.idx][1] - self.bytes_offset[self.rank] < lim + SEPARATOR_BYTES
        ):
            assert (
                self.bytes_counter - SEPARATOR_BYTES
                < self.bytes_ranges[self.idx][0] - self.bytes_offset[self.rank]
                < lim
            ), f"{self.bytes_counter=} > {self.bytes_ranges[self.idx][0] - self.bytes_offset[self.rank]}"
            ranges.append(self.bytes_ranges[self.idx])
            self.idx += 1

            if self.idx == len(self.bytes_ranges):
                self.exhausted_ranges = True
                break
        return ranges

    def normalize_range(self, x, n_bytes):
        assert (
            self.bytes_offset[self.rank] < x[0] < x[1] < self.bytes_offset[self.rank + 1] + SEPARATOR_BYTES
        ), f"{self.bytes_offset[self.rank]=}, {x[0]} {x[1]} {self.bytes_offset[self.rank + 1]=}"

        offset = self.bytes_offset[self.rank] + self.bytes_counter
        a, b = x[0] - offset, x[1] - offset
        assert all([a > -SEPARATOR_BYTES, b > 0]), f"byte_a={a}, byte_b={b}"
        assert all([a < n_bytes, b < n_bytes + SEPARATOR_BYTES]), f"byte_a={a}, byte_b={b}"
        a = max(SEPARATOR_BYTES, a)
        b = min(n_bytes, b)

        # TODO IMPROVE
        if (b - a) % 2 != 0:
            if b == n_bytes:
                a += 1
            else:
                b += 1

        return a, b

    def remove_duplicate(self, doc, bytes_content):
        n_bytes = len(bytes_content)
        duplicates_ranges = self.get_range(n_bytes)
        duplicates = []
        for dup_range in duplicates_ranges:
            byte_a, byte_b = self.normalize_range(dup_range, n_bytes)
            dup_sentence = self.tokenizer.decode(np.frombuffer(bytes_content[byte_a:byte_b], dtype=np.uint16).tolist())
            duplicates.append(dup_sentence)

        if duplicates:
            text = doc.content
            # TODO improve
            for d in duplicates:
                text = text.replace(d, "")
            doc.content = text

        self.bytes_counter += len(bytes_content)

        if len(word_tokenize(doc.content, get_language(doc))) < self.min_doc_words:
            return False

        return True

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        self.reset()
        self.rank = rank
        self.read_bytes_offset()

        sequence_file, size_file = self.get_all_files(rank=self.rank, world_size=world_size)
        # data is given only during tests.
        if not data:
            data = self.read_files_shard(self.data_folder.get_files_shard(self.rank, world_size))

        for doc, doc_content in zip(data, sequence_reader(sequence_file, size_file)):
            with self.stats.time_manager:
                # we check that the two generators are synced.
                assert doc.content == self.tokenizer.decode(
                    read_bytes(doc_content)
                ), f"{doc.content}\n\n{self.tokenizer.decode(read_bytes(doc_content))}"
                to_yield = self.remove_duplicate(doc, doc_content)
            if to_yield:
                self.stats.doc_len.update(len(doc.content))
                yield doc

        # we check bytes counter matches with the offset of the following rank
        assert (
            self.bytes_counter == self.bytes_offset[rank + 1] - self.bytes_offset[rank]
        ), f"got {self.bytes_counter=}, expected = {self.bytes_offset[rank + 1] - self.bytes_offset[rank]}"
        assert self.exhausted_ranges, "A duplicate range has not been used!"
