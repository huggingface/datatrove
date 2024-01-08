"""Here we implement exact substrings as suggested by https://arxiv.org/pdf/2107.06499.pdf.
We use suffix array to deduplicate exact substrings above a minimum threshold. We will take the code tom build the
actual suffix array and find duplicates form the GitHub page of "Deduplicating Training Data Makes Language Models
Better".
TLDR
1) DatasetToSequence map 1 file into a sequence S. With unique separators at the beginning of each doc. It also saves
   a second file with the bytes offset of where each individual doc begins.
2) MergeSequences all sequences into a big single sequence. it saves the bytes offset per file.

 ... call deduplicate-text-datasets scripts
     in particular `cargo run self-similar ...` and `cargo run self-similar` need to be called

3) DedupReader reads docs and ranges at the same time and remove duplicates.

"""
import struct
from typing import Generator, Literal

import numpy as np
import tokenizers
from loguru import logger
from nltk.tokenize import word_tokenize

from datatrove.io import BaseInputDataFile, BaseInputDataFolder, BaseOutputDataFolder
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep
from datatrove.pipeline.readers import JsonlReader

from .utils import ExtensionHelperES as EH


SEPARATOR_BYTES = 12


def prepare_doc(tokenizer, doc: str, rank: int, doc_id: int):
    tokens = tokenizer.encode(doc).ids
    tokens = np.array(tokens, dtype=np.uint16)
    b_doc = b"\xff\xff" + struct.pack("<I", doc_id) + b"\xff\xff" + struct.pack("<I", rank) + tokens.tobytes()
    return b_doc


class DatasetToSequence(PipelineStep):
    """STAGE 1
    Creates a sequence of all docs pre-prepended by a unique separator. It also saves a second file with the
    bytes offset of where each individual doc begins.
    """

    type = "🫂 - DEDUP"
    name = "🪞 - exact-substrings stage 1"

    def __init__(self, output_folder=BaseOutputDataFolder, tokenizer_name: str = "gpt2"):
        """Args:
        output_folder: folder where sequences are saved
        tokenizer_name: name of tokenizer as in HF tokenizers.
        """
        super().__init__()
        self.output_folder = output_folder
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_name)

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def save_sizes(self, doc_lens: list[int], rank: int):
        f_lens = self.output_folder.open(f"{rank:05d}{EH.stage_1_sequence_size}", mode="wb")
        f_lens.write(struct.pack("Q" * len(doc_lens), *doc_lens))

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        doc_lens = []
        f_sequence = self.output_folder.open(f"{rank:05d}{EH.stage_1_sequence}", mode="wb")
        for i, doc in enumerate(data):
            with self.stats.time_stats:
                b_doc = prepare_doc(tokenizer=self.tokenizer, doc=doc.text, rank=rank, doc_id=i)
                doc_lens.append(len(b_doc))
                f_sequence.write(b_doc)

        assert i < 2**32, "doc ID overflow"
        assert i + 1 == len(doc_lens), f"{i=} but {len(doc_lens)=}"

        self.save_sizes(doc_lens, rank)
        self.output_folder.close()


class MergeSequences(PipelineStep):
    """STAGE 2
    It merges all the sequences from stage 1 into a big sequence. It saves a file with the cumulative bytes offset
    of every single sequence.
    """

    type = "🫂 - DEDUP"
    name = "🪞 - exact-substrings stage 2"

    def __init__(
        self,
        input_folder: BaseInputDataFolder,
        output_folder: BaseOutputDataFolder,
        tasks_stage_1: int,
        bytes_per_batch: int = int(500e6),
    ):
        """Args:
        input_folder: folder where sequences were saved in stage 1
        output_folder: folder where the big sequence will be saved
        tasks_stage_1: number of tasks used in stage 1
        bytes_per_batch: number of bytes read per sequence
        """
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.tasks_stage_1 = tasks_stage_1
        self.bytes_per_batch = bytes_per_batch

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        bytes_per_sequence = [0]
        with self.stats.time_stats:
            assert world_size == 1, f"{world_size=} can't be greater than 1!"
            all_files: list[BaseInputDataFile] = self.input_folder.list_files(extension=EH.stage_1_sequence)
            assert len(all_files) == self.tasks_stage_1
            f_sequence = self.output_folder.open(f"dataset{EH.stage_2_big_sequence}", mode="wb")
            for file in all_files:
                len_sequence = 0
                with file.open(binary=True) as f:
                    while True:
                        sequence = f.read(self.bytes_per_batch)
                        f_sequence.write(sequence)
                        len_sequence += len(sequence)
                        if len(sequence) != self.bytes_per_batch:
                            break
                    bytes_per_sequence.append(bytes_per_sequence[-1] + len_sequence)

            f_bytes = self.output_folder.open(f"bytes_offsets{EH.stage_2_bytes_offset}", mode="wb")
            f_bytes.write(np.array([bytes_per_sequence], np.uint32).tobytes())
            self.output_folder.close()


def read_bytes(x):
    # 4 bytes for rank + 4 bytes for  2 * b"\xff\xff" + 4 bytes for doc_id
    return np.frombuffer(x[SEPARATOR_BYTES:], dtype=np.uint16).tolist()


def sequence_reader(file: BaseInputDataFile, size_file: BaseInputDataFile) -> Generator[list, None, None]:
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
        compression: Literal["guess", "gzip", "zst"] | None = "guess",
        tokenizer_name: str = "gpt2",
        min_doc_words: int = 50,
    ):
        super().__init__(data_folder=data_folder, compression=compression)
        self.sequence_folder = sequence_folder
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_name)
        self.min_doc_words = min_doc_words
        self.sequence_bytes_offset = None
        self.dup_ranges = None
        self.rank = None
        self.exhausted_ranges = False
        self.bytes_counter = 0
        self.range_idx = 0

    def reset(self):
        self.bytes_counter = 0
        self.range_idx = 0
        self.exhausted_ranges = False
        self.sequence_bytes_offset = None
        self.dup_ranges = None
        self.rank = None

    def get_sequence_bytes_offset(self):
        offset_array_file: BaseInputDataFile = self.sequence_folder.list_files(extension=EH.stage_2_bytes_offset)[0]
        with offset_array_file.open(binary=True) as f:
            offset_array = f.read()
        self.sequence_bytes_offset = np.frombuffer(offset_array, dtype=np.uint32)
        logger.info(f"{self.rank=}, -> {self.sequence_bytes_offset[self.rank]=}")

    def get_bytearange(self, bytes_range_file: BaseInputDataFile):
        with bytes_range_file.open(binary=False) as f:
            dup_ranges = f.read()

        dup_ranges = dup_ranges.split("\n")
        for i, x in enumerate(dup_ranges):
            if x == "out":
                break

        # remove lines until out and remove last empty value
        dup_ranges = dup_ranges[i + 1 : -1]

        rank_dup_ranges = []
        for br in dup_ranges:
            a, b = br.split(" ")
            a, b = int(a), int(b)
            if b > self.sequence_bytes_offset[self.rank + 1] + SEPARATOR_BYTES:
                break
            if b > self.sequence_bytes_offset[self.rank] + SEPARATOR_BYTES:
                a, b = a - self.sequence_bytes_offset[self.rank], b - self.sequence_bytes_offset[self.rank]
                rank_dup_ranges.append((a, b))
        self.dup_ranges = rank_dup_ranges

    def get_all_files(self, rank: int, world_size: int):
        self.get_sequence_bytes_offset()
        sequence_file = self.sequence_folder.get_files_shard(rank, world_size, extension=EH.stage_1_sequence)
        docs_sizes_file = self.sequence_folder.get_files_shard(rank, world_size, extension=EH.stage_1_sequence_size)
        byte_range_file = self.sequence_folder.list_files(extension=EH.stage_3_bytes_ranges)

        assert all(
            [len(sequence_file) == 1, len(docs_sizes_file) == 1, len(byte_range_file) == 1]
        ), f"Need to run with n_tasks = n_files. {len(sequence_file)=}, {len(sequence_file)=}, {len(byte_range_file)=}"
        sequence_file, docs_sizes_file, byte_range_file = sequence_file[0], docs_sizes_file[0], byte_range_file[0]

        self.get_bytearange(byte_range_file)
        return sequence_file, docs_sizes_file

    def normalize_range(self, a, b, bytes_len):
        a, b = a - self.bytes_counter, b - self.bytes_counter
        a = max(SEPARATOR_BYTES, a)
        b = min(bytes_len, b)
        assert (
            SEPARATOR_BYTES <= a < b <= bytes_len
        ), f"{SEPARATOR_BYTES=} < {a=} < {b=} < {bytes_len=} is NOT satisfied"

        # TODO IMPROVE
        if (b - a) % 2 != 0:
            if b == bytes_len:
                a += 1
            else:
                b += 1

        return a, b

    def get_duplicate_range(self, bytes_len: int):
        """Ranges produced by deduplicate-text-dataset can fall in one of the following 4 categories

                   left    )  A   *    B    *       A --> *, idx <-- idx + 1
                   centre  )  *   A    B    *       idx <-- idx + 1
                   right   )  *   A    *    B       B --> *
                   outside )  A   *    *    B       A --> *, B --> *

        * is self.bytes_counter
        * is upper_limit =  self.bytes_counter + bytes_len

        """
        ranges = []
        upper_limit = self.bytes_counter + bytes_len + SEPARATOR_BYTES

        if self.exhausted_ranges:
            return ranges

        while True:
            a, b = self.dup_ranges[self.range_idx][0], self.dup_ranges[self.range_idx][1]

            left = a < self.bytes_counter and self.bytes_counter + SEPARATOR_BYTES < b <= upper_limit
            centre = self.bytes_counter <= a < b <= upper_limit
            right = self.bytes_counter <= a < upper_limit - SEPARATOR_BYTES and upper_limit < b
            outside = a < self.bytes_counter < upper_limit < b

            if not any([left, centre, right, outside]):
                break

            assert sum([left, centre, right, outside]) == 1, f"{left=}, {centre=}, {right=}, {outside=}"

            if left:
                self.range_idx += 1
                a = self.bytes_counter
            if centre:
                self.range_idx += 1
            if right:
                ranges.append(self.normalize_range(a, upper_limit, bytes_len))
                break
            if outside:
                ranges.append(self.normalize_range(self.bytes_counter, upper_limit, bytes_len))
                break

            ranges.append(self.normalize_range(a, b, bytes_len))

            if self.range_idx == len(self.dup_ranges):
                self.exhausted_ranges = True
                break

        return ranges

    def remove_duplicate(self, doc, bytes_content):
        n_bytes = len(bytes_content)
        duplicates_ranges = self.get_duplicate_range(n_bytes)
        duplicates = []
        for byte_a, byte_b in duplicates_ranges:
            dup_sentence = self.tokenizer.decode(np.frombuffer(bytes_content[byte_a:byte_b], dtype=np.uint16).tolist())
            duplicates.append(dup_sentence)

        if duplicates:
            text = doc.text
            # TODO improve
            for d in duplicates:
                text = text.replace(d, "")
            doc.text = text

        self.bytes_counter += len(bytes_content)

        if len(word_tokenize(doc.text)) < self.min_doc_words:
            return False

        return True

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        self.reset()
        self.rank = rank
        # loads the sequence file from stage 1, the size file from stage 1 and the bytearange file.
        sequence_file, size_file = self.get_all_files(rank=self.rank, world_size=world_size)
        # data is given only during tests.
        if not data:
            data = self.read_files_shard(self.data_folder.get_files_shard(self.rank, world_size))
        # data is still useful for the metadata lost in the sequence format.
        for doc, doc_content in zip(data, sequence_reader(sequence_file, size_file)):
            with self.stats.time_stats:
                # We check that the two generators are synced, meaning the docs sizes bytes are correct.
                assert doc.text == self.tokenizer.decode(
                    read_bytes(doc_content), skip_special_tokens=False
                ), f"{doc.text}\n\n{self.tokenizer.decode(read_bytes(doc_content))}"
                to_yield = self.remove_duplicate(doc, doc_content)
            if to_yield:
                self.update_doc_stats(doc)
                yield doc

        # we check bytes counter matches with the offset of the following rank
        assert (
            self.bytes_counter == self.sequence_bytes_offset[rank + 1] - self.sequence_bytes_offset[rank]
        ), f"got {self.bytes_counter=}, expected = {self.sequence_bytes_offset[rank + 1] - self.sequence_bytes_offset[rank]}"
        assert self.exhausted_ranges, "One or more duplicate ranges have not been used"
