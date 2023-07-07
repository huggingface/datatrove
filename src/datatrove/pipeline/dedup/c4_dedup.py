"""
'To deduplicate the data set, we discarded all but one of any three-sentence span
occurring more than once in the data set.'

from: https://jmlr.org/papers/volume21/20-074/20-074.pdf (C4)

# get hashes for each doc and write them down

"""

import struct
from dataclasses import dataclass
from queue import PriorityQueue

from nltk.tokenize import sent_tokenize, word_tokenize

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import InputDataFolder, OutputDataFolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints

from .utils import merge_docs, simplify_content, str_hash


@dataclass
class HashSig:
    hash_value: int
    doc_id: int
    sent_id: int
    file_id: int = None

    # priority queue accepts anything that is sortable
    def __lt__(self, other):
        return (self.hash_value, self.file_id, self.doc_id, self.sent_id) < (
            other.hash_value,
            other.file_id,
            other.doc_id,
            other.sent_id,
        )


class DuFile:
    def __init__(self, file_name: str, mode: str):
        self.mode = mode
        self.file_name = file_name
        self.file_handler = open(self.file_name, self.mode)

    def write(self, hs: HashSig):
        if self.mode == "rb":
            raise ValueError
        self.file_handler.write(struct.pack("<I", hs.doc_id))
        self.file_handler.write(struct.pack("<H", hs.sent_id))

    def read(self):
        if self.mode == "ab":
            raise ValueError
        x = []
        for k, b in [("I", 4), ("H", 2)]:
            by = self.file_handler.read(b)
            if not by:
                self.file_handler.close()
                return None
            x.append(struct.unpack(f"<{k}", by)[0])
        return tuple(x)

    def read_all(self):
        all_x = []
        while True:
            x = self.read()
            if not x:
                break
            all_x.append(x)
        return all_x


class SigFile:
    def __init__(self, file_name: str, mode: str):
        self.file_name = file_name
        self.file_handler = open(self.file_name, mode=mode)

    def write(self, hs: HashSig):
        self.file_handler.write(struct.pack("<Q", hs.hash_value))
        self.file_handler.write(struct.pack("<I", hs.doc_id))
        self.file_handler.write(struct.pack("<H", hs.sent_id))

    def read(self, file_id: int) -> None | HashSig:
        x = []
        for k, b in [("Q", 8), ("I", 4), ("H", 2)]:
            by = self.file_handler.read(b)
            if not by:
                self.file_handler.close()
                return None
            x.append(struct.unpack(f"<{k}", by)[0])
        x.append(file_id)
        return HashSig(*x)


class C4DedupSignature(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🫧 C4 stage 1"

    def __init__(self, output_folder: OutputDataFolder, n_sentences: int = 3, stage_2_workers: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.output_folder = output_folder
        self.n_sentences = n_sentences
        self.stage_2_workers = stage_2_workers
        self.signatures = []

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def save_hashes(self, rank: int):
        self.signatures = sorted(self.signatures, key=lambda x: x.hash_value)

        f = self.output_folder.get_file(f"{rank:05d}.c4_sig", lambda x: open(x, "wb"))
        for hs in self.signatures:
            f.file_handler.write(struct.pack("<Q", hs.hash_value))
            f.file_handler.write(struct.pack("<I", hs.doc_id))
            f.file_handler.write(struct.pack("<H", hs.sent_id))

    def get_hashes(self, doc: Document, doc_idx: int) -> list[None] | list[HashSig]:
        sentences = sent_tokenize(doc.content)
        if len(sentences) < self.n_sentences:
            return []

        sentences_tokens = [simplify_content(sent) for sent in sentences]
        n_sent_grams: list = [
            " ".join(sentences_tokens[i : i + self.n_sentences])
            for i in range(len(sentences_tokens) - self.n_sentences + 1)
        ]
        hashes = [
            HashSig(
                hash_value=str_hash(n_sent_gram),
                doc_id=doc_idx,
                sent_id=sentence_idx,
            )
            for sentence_idx, n_sent_gram in enumerate(n_sent_grams)
        ]

        return hashes

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc_idx, doc in enumerate(data):
            self.stat_update(StatHints.total)
            self.signatures.extend(self.get_hashes(doc, doc_idx))
        self.save_hashes(rank)
        self.output_folder.close()


class C4CreateDedups(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🫧 C4 stage 2"

    def __init__(self, data_folder: InputDataFolder, output_folder: OutputDataFolder, **kwargs):
        super().__init__(**kwargs)
        self.data_folder = data_folder
        self.output_folder = output_folder

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        priority_queue = PriorityQueue()
        # todo waiting for new IO
        in_file_handles = [
            SigFile(file_name=f"/Users/alessandrocappelli/PycharmProjects/datatrove/c4/{i:05d}.c4_sig", mode="rb")
            for i in range(4)
        ]

        for i, fh in enumerate(in_file_handles):
            priority_queue.put(fh.read(i))

        last = None
        while not priority_queue.empty():
            v: HashSig = priority_queue.get()
            if last == v.hash_value:
                f = self.output_folder.get_file(f"{v.file_id:05d}.c4_dup", lambda x: open(x, "wb"))
                f.file_handler.write(struct.pack("<I", v.doc_id))
                f.file_handler.write(struct.pack("<H", v.sent_id))
            last = v.hash_value
            new_v = in_file_handles[v.file_id].read(v.file_id)
            if new_v:
                priority_queue.put(new_v)


class C4Filter(PipelineStep):
    type = "🫂 - DEDUP"
    name = "🫧 C4 stage 3"

    def __init__(
        self,
        data_folder: InputDataFolder,
        min_doc_words: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_folder = data_folder
        self.min_doc_words = min_doc_words

    def filter(self, doc: Document, du_lines: set = None):
        sentences = sent_tokenize(doc.content)
        doc.content = " ".join([sent for idx, sent in enumerate(sentences) if not du_lines or idx not in du_lines])
        if len(word_tokenize(doc.content)) > 50:
            return True
        return False

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        step method for Filters.
        Drops documents that if .filter() is False

        @param datapipe: input DocumentsPipeline
        @return: DocumentsPipeline
        """
        du_file = merge_docs(
            sorted(
                # todo waiting for new IO
                DuFile(
                    file_name=f"/Users/alessandrocappelli/PycharmProjects/datatrove/c4/{rank:05d}.c4_dup", mode="rb"
                ).read_all()
            )
        )

        for idx, doc in enumerate(data):
            self.stat_update(StatHints.total)
            with self.time_stats_manager:
                is_kept = self.filter(doc, du_lines=du_file.get(idx))
            if is_kept:
                yield doc
