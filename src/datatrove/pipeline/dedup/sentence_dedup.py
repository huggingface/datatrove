"""
'To deduplicate the data set, we discarded all but one of any three-sentence span
occurring more than once in the data set.'

from: https://jmlr.org/papers/volume21/20-074/20-074.pdf (C4)

# get hashes for each doc and write them down

"""
import heapq
import struct
from dataclasses import dataclass
from typing import Generator

from nltk.tokenize import sent_tokenize, word_tokenize

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import BaseInputDataFolder, BaseOutputDataFolder, InputDataFile
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints

from .utils import ExtensionHelperSD, merge_docs, simplify_content, str_hash


@dataclass
class HashSig:
    hash_value: int
    doc_id: int
    sent_id: int
    file_id: int = None

    # priority queue accepts anything that is sortable
    def __lt__(self, other) -> bool:
        return (self.hash_value, self.file_id, self.doc_id, self.sent_id) < (
            other.hash_value,
            other.file_id,
            other.doc_id,
            other.sent_id,
        )


class SentenceDedupSignature(PipelineStep):
    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 1"

    def __init__(self, output_folder: BaseOutputDataFolder, n_sentences: int = 3, **kwargs):
        """

        :param output_folder: folder where signatures are saved
        :param n_sentences: n_sentences where duplicates are checked.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_folder = output_folder
        self.n_sentences = n_sentences
        self.signatures = []

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def save_hashes(self, rank: int):
        self.signatures.sort()

        f = self.output_folder.open(f"{rank:05d}{ExtensionHelperSD.stage_1_signature}", mode="wb")
        for hs in self.signatures:
            f.file_handler.write(struct.pack("<Q", hs.hash_value))
            f.file_handler.write(struct.pack("<I", hs.doc_id))
            f.file_handler.write(struct.pack("<H", hs.sent_id))

    def get_hashes(self, doc: Document, doc_idx: int) -> list[None] | list[HashSig]:
        # todo use language id metadata in sent_tokenize
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
        """

        :param data:
        :param rank:
        :param world_size:
        :return:

        SentenceDedupSignature creates a signature for each document. Each HashSig has n hash, the doc id and the
        sentence idx. Before saving them the hashes are sorted.

        """
        self.signatures = []
        for doc_idx, doc in enumerate(data):
            with self.stats.time_manager:
                self.stat_update(StatHints.total)
                self.signatures.extend(self.get_hashes(doc, doc_idx))
        self.save_hashes(rank)
        self.output_folder.close()


def read_sigs(file: InputDataFile, file_id: int) -> Generator[HashSig, None, None]:
    with file.open(binary=True) as f:
        while True:
            x = {}
            for t, b, k in [
                ("Q", struct.calcsize("Q"), "hash_value"),
                ("I", struct.calcsize("I"), "doc_id"),
                ("H", struct.calcsize("H"), "sent_id"),
            ]:
                by = f.read(b)
                if not by:
                    return
                x[k] = struct.unpack(f"<{t}", by)[0]
            yield HashSig(file_id=file_id, **x)


class SentenceFindDedups(PipelineStep):
    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 2"

    def __init__(self, data_folder: BaseInputDataFolder, output_folder: BaseOutputDataFolder, **kwargs):
        super().__init__(**kwargs)
        self.data_folder = data_folder
        self.output_folder = output_folder

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        """

        :param data:
        :param rank:
        :param world_size:
        :return:

        SentenceFindDedups runs on a single worker. It reads all the signatures from the previous step and load them
        in a priority queue to check for duplicates. If a duplicate is found its document id and sentence id are saved.
        """
        assert world_size == 1, "SentenceFindDedups can only run on a single worker."
        files_with_duplicates = set()
        with self.stats.time_manager:
            sig_files = self.data_folder.list_files(ExtensionHelperSD.stage_1_signature)
            sig_readers = [read_sigs(file, file_i) for file_i, file in enumerate(sig_files)]

            pq = [next(sig_reader) for sig_reader in sig_readers]
            heapq.heapify(pq)

            last = None
            while pq:
                v: HashSig = heapq.heappop(pq)
                if last == v.hash_value:
                    f = self.output_folder.open(f"{v.file_id:05d}{ExtensionHelperSD.stage_2_duplicates}", mode="wb")
                    f.file_handler.write(struct.pack("<I", v.doc_id))
                    f.file_handler.write(struct.pack("<H", v.sent_id))
                    files_with_duplicates.add(v.file_id)
                last = v.hash_value
                new_v = next(sig_readers[v.file_id], None)

                if new_v:
                    heapq.heappush(pq, new_v)

        for i in range(len(sig_files)):
            if i not in files_with_duplicates:
                # empty files as the next stage expects 1 file per task
                self.output_folder.open(f"{i:05d}{ExtensionHelperSD.stage_2_duplicates}", mode="wb")
        self.output_folder.close()


def read_duplicates(file: InputDataFile) -> Generator[tuple, None, None]:
    with file.open(binary=True) as f:
        while True:
            x = []
            for (
                t,
                b,
            ) in [("I", struct.calcsize("I")), ("H", struct.calcsize("H"))]:
                by = f.read(b)
                if not by:
                    return
                x.append(struct.unpack(f"<{t}", by)[0])
            yield tuple(x)  # (doc_id, sent_id) pairs


class SentenceDedupFilter(PipelineStep):
    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 3"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        n_sentences: int = 3,
        min_doc_words: int = 50,
        **kwargs,
    ):
        """

        :param data_folder: data folder to get duplicate files.
        :param min_doc_words: min amount of words for each document
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.data_folder: BaseInputDataFolder = data_folder
        self.n_sentences = n_sentences
        self.min_doc_words = min_doc_words

    def filter(self, doc: Document, du_lines: set = None):
        if not du_lines:
            return True
        sentences = sent_tokenize(doc.content)
        filtered_sentences = [sent for idx, sent in enumerate(sentences) if not du_lines or idx not in du_lines]
        if len(filtered_sentences) < len(sentences):
            self.stat_update("removed_sentences", len(sentences) - len(filtered_sentences))
        self.stat_update("original_sentences", len(sentences))
        doc.content = " ".join(filtered_sentences).strip()
        if len(word_tokenize(doc.content)) > self.min_doc_words:
            return True
        return False

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        step method for Filters.
        Drops documents that if .filter() is False

        @param datapipe: input DocumentsPipeline
        @return: DocumentsPipeline

        SentenceDedupFilter reads a DocumentPipeline and removes duplicated sentences found at stage 2
        """
        files = self.data_folder.get_files_shard(rank, world_size, extension=ExtensionHelperSD.stage_2_duplicates)
        assert len(files) == 1, (
            f"n_files / n_tasks should be equal to n_workers, instead {len(files)=}\n{files}.\n"
            f"{world_size=} {rank}"
        )

        du_file = merge_docs(sorted(read_duplicates(files[0])), self.n_sentences)
        for idx, doc in enumerate(data):
            self.stat_update(StatHints.total)
            with self.stats.time_manager:
                is_kept = self.filter(doc, du_lines=du_file.get(idx))
            if is_kept:
                self.stats.doc_len.update(len(doc.content))
                yield doc
