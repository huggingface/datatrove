"""
'To deduplicate the data set, we discarded all but one of any three-sentence span
occurring more than once in the data set.'

from: https://jmlr.org/papers/volume21/20-074/20-074.pdf (C4)

# get hashes for each doc and write them down

"""

import hashlib
import os
from queue import PriorityQueue

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints

from .utils import simplify_content


def split_hash_info(hash_info: str, file_idx: int) -> tuple[str, ..., int]:
    x = tuple(hash_info.split(",,")) + (file_idx,)
    print(x)
    assert len(x) == 4, f"info len = {len(x)}. Something went terribly wrong!\n{x}"
    return x


class C4DedupSignature(PipelineStep):
    type = "DEDUP"
    name = "C4 stage 2"

    def __init__(self, n_sentences: int = 3, stage_2_workers: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.n_sentences = n_sentences
        self.stage_2_workers = stage_2_workers
        self.signatures = []

    def save_hashes(self, rank: int):
        self.signatures = sorted(self.signatures, key=lambda x: x[0])
        # TODO split into MAX / stage_2_workers files if stage_2_workers > 1
        with open(f"c4_sig_{rank}", "wb") as f:
            for sig in self.signatures:
                f.write(bytes(f"{sig[0]},,{sig[1]},,{sig[2]}\n", encoding="utf8"))

    def get_hashes(self, doc: Document, doc_idx: int) -> list[None] | list[tuple[bytes, int, int]]:
        sentences = sent_tokenize(doc.content)
        if len(sentences) < self.n_sentences:
            return []

        sentences_tokens = [self.tokenizer.encode(simplify_content(sent)) for sent in sentences]
        n_sent_grams: list = [
            sum(sentences_tokens[i : i + self.n_sentences], []) for i in range(len(sentences_tokens))
        ]
        hashes = [
            (
                hashlib.sha1(f"{n_sent_gram}".encode("utf-8")).digest(),
                doc_idx,
                sentence_idx,
            )
            for sentence_idx, n_sent_gram in enumerate(n_sent_grams)
        ]

        return hashes

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc_idx, doc in enumerate(data):
            self.stat_update(StatHints.total)
            self.signatures.extend(self.get_hashes(doc, doc_idx))
        self.save_hashes(rank)


class C4CreateDedups(PipelineStep):
    type = "DEDUP"
    name = "C4 stage 2"

    def __init__(self, stage_1_workers, **kwargs):
        super().__init__(**kwargs)
        self.stage_1_workers = stage_1_workers
        self._pq = None  # classes give problem to deepcopy

    @property
    def priority_queue(self):
        if not self._pq:
            self._pq = PriorityQueue()
        return self._pq

    def write_dedup(self):
        raise NotImplementedError

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        assert all(os.path.isfile(f"c4_sig_{i}") for i in range(self.stage_1_workers))
        file_handles = [open(f"c4_sig_{i}", "r") for i in range(self.stage_1_workers)]
        # TODO fix readline() reads even when file is empty
        for i, fh in enumerate(file_handles):
            self.priority_queue.put(split_hash_info(fh.readline(), i))
        last = -1
        while not self.priority_queue.empty():
            v = self.priority_queue.get()
            if last == v:
                self.write_dedup()
            last = v
            self.priority_queue.put(split_hash_info(file_handles[v[3]].readline(), v[3]))


class C4Filter(PipelineStep):
    name = "🥇 Gopher Quality"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._dedup_file = None

    @property
    def dedup_file(self, rank):
        if not self._dedup_file:
            self._dedup_file = open(f"dedup_file_{rank}", "r")
        return self._dedup_file

    def filter(self, doc: Document, idx):
        raise NotImplementedError

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        step method for Filters.
        Drops documents that if .filter() is False

        @param datapipe: input DocumentsPipeline
        @return: DocumentsPipeline
        """
        dup_info = None
        for idx, doc in enumerate(data):
            self.stat_update(StatHints.total)
            with self.time_stats_manager:
                if not dup_info or dup_info.index < idx:
                    dup_info = self.dedup_file.readline()
                if dup_info.index != idx:
                    is_kept = True
                else:
                    is_kept = self.filter(doc, dup_info)
            if is_kept:
                yield doc
