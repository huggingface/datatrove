"""'To deduplicate the data set, we discarded all but one of any three-sentence span
occurring more than once in the data set.'

from: https://jmlr.org/papers/volume21/20-074/20-074.pdf (C4)

# get hashes for each doc and write them down

"""
import contextlib
import dataclasses
import heapq
import struct
from dataclasses import dataclass
from typing import Generator

from loguru import logger
from nltk import load, ngrams
from nltk.tokenize import sent_tokenize, word_tokenize

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import BaseInputDataFile, BaseInputDataFolder, BaseOutputDataFolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints

from ..writers.disk_base import DiskWriter
from .utils import ExtensionHelperSD, merge_docs, read_tuples_from_file, simplify_text, str_hash


@dataclass(order=True)
class HashSig:
    # this also determines the sorting order
    # hash_value needs to come first as that's what we match on
    # file_id should come after doc_id so that hashes from the index (sent_id=doc_id=-1) come up first
    hash_value: int
    doc_id: int
    file_id: int = None
    sent_id: int = None

    def is_from_index(self):
        return self.doc_id == self.sent_id == -1


class SentenceDedupSignature(PipelineStep):
    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 1"

    def __init__(self, output_folder: BaseOutputDataFolder, n_sentences: int = 3):
        """Args:
        output_folder: folder where signatures are saved
        n_sentences: n_sentences where duplicates are checked.
        kwargs
        """
        super().__init__()
        self.output_folder = output_folder
        self.n_sentences = n_sentences

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.output_folder.set_lock(up_lock)

    def save_hashes(self, rank: int, signatures):
        signatures.sort()

        f = self.output_folder.open(f"{rank:05d}{ExtensionHelperSD.stage_1_signature}", mode="wb")
        for hs in signatures:
            f.write(struct.pack("<Q", hs.hash_value))
            f.write(struct.pack("<I", hs.doc_id))
            f.write(struct.pack("<H", hs.sent_id))

    def get_hashes(self, doc: Document, doc_idx: int) -> list[None] | list[HashSig]:
        # todo use language id metadata in sent_tokenize
        sentences = sent_tokenize(doc.text)
        if len(sentences) < self.n_sentences:
            return []

        sentences_tokens = [simplify_text(sent) for sent in sentences]
        n_sent_grams: list = [" ".join(x) for x in ngrams(sentences_tokens, self.n_sentences)]
        hashes = [
            HashSig(
                hash_value=str_hash(n_sent_gram),
                doc_id=doc_idx,
                sent_id=sentence_idx,
            )
            for sentence_idx, n_sent_gram in enumerate(n_sent_grams)
        ]

        return hashes

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        """Args:
            data
            rank
            world_size

        Returns:

        SentenceDedupSignature creates a signature for each document. Each HashSig has n hash, the doc id and the
        sentence idx. Before saving them the hashes are sorted.

        """
        signatures = []
        for doc_idx, doc in enumerate(data):
            with self.stats.time_stats:
                self.stat_update(StatHints.total)
                signatures.extend(self.get_hashes(doc, doc_idx))
        self.save_hashes(rank, signatures)
        self.output_folder.close()


def read_sigs(file: BaseInputDataFile, file_id: int, index_file: bool = False) -> Generator[HashSig, None, None]:
    if index_file:
        # only read hashes
        for (hash,) in read_tuples_from_file(file, "Q"):
            yield HashSig(hash_value=hash, doc_id=-1, file_id=file_id, sent_id=-1)
    else:
        for hash, doc_id, sent_id in read_tuples_from_file(file, "Q", "I", "H"):
            yield HashSig(file_id=file_id, hash_value=hash, doc_id=doc_id, sent_id=sent_id)


class SentenceFindDedups(PipelineStep):
    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 2"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        output_folder: BaseOutputDataFolder,
        index_folder: BaseInputDataFolder = None,
        only_dedup_in_index: bool = True,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.index_folder = index_folder
        self.only_dedup_in_index = only_dedup_in_index

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        """Args:
            data
            rank
            world_size

        Returns:

        SentenceFindDedups runs on a single worker. It reads all the signatures from the previous step and load them
        in a priority queue to check for duplicates. If a duplicate is found its document id and sentence id are saved.
        """
        assert world_size == 1, "SentenceFindDedups can only run on a single worker."
        files_with_duplicates = set()
        with self.stats.time_stats:
            sig_files = self.data_folder.list_files(extension=ExtensionHelperSD.stage_1_signature)
            sig_readers = [read_sigs(file, file_i) for file_i, file in enumerate(sig_files)]
            index_files = self.index_folder.list_files() if self.index_folder else None
            if index_files:
                logger.info(f"Found index file(s): {', '.join([file.relative_path for file in index_files])}")
                sig_readers.extend(
                    [
                        read_sigs(file, len(sig_readers) + file_i, index_file=True)
                        for file_i, file in enumerate(index_files)
                    ]
                )

            pq = [next(sig_reader) for sig_reader in sig_readers]
            heapq.heapify(pq)

            last: HashSig | None = None
            while pq:
                v: HashSig = heapq.heappop(pq)
                if (
                    last and last.hash_value == v.hash_value and not v.is_from_index()
                ):  # we never want to match samples from the index itself
                    f = self.output_folder.open(f"{v.file_id:05d}{ExtensionHelperSD.stage_2_duplicates}", mode="wb")
                    # the previous one we are matching against is part of the index
                    # OR there are no index files
                    # OR we are also matching within the main dataset
                    if last.is_from_index() or not index_files or not self.only_dedup_in_index:
                        f.write(struct.pack("<I", v.doc_id))
                        f.write(struct.pack("<H", v.sent_id))
                        files_with_duplicates.add(v.file_id)
                last = v
                new_v = next(sig_readers[v.file_id], None)

                if new_v:
                    heapq.heappush(pq, new_v)

        for i in range(len(sig_files)):
            if i not in files_with_duplicates:
                # empty files as the next stage expects 1 file per task
                self.output_folder.open(f"{i:05d}{ExtensionHelperSD.stage_2_duplicates}", mode="wb")
        self.output_folder.close()


def read_duplicates(file: BaseInputDataFile) -> Generator[tuple, None, None]:
    yield from read_tuples_from_file(file, "I", "H")  # (doc_id, sent_id) pairs


class SentenceDedupFilter(PipelineStep):
    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 3"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        n_sentences: int = 3,
        min_doc_words: int = 50,
        exclusion_writer: DiskWriter = None,
    ):
        """Args:
        data_folder: data folder to get duplicate files.
        min_doc_words: min amount of words for each document
        """
        super().__init__()
        self.data_folder: BaseInputDataFolder = data_folder
        self.n_sentences = n_sentences
        self.min_doc_words = min_doc_words
        self._tokenizer = load("tokenizers/punkt/english.pickle")
        self.exclusion_writer = exclusion_writer

    def remove_dup_sentences(self, doc: Document, du_lines: set = None) -> (str, str):
        if not du_lines:
            return doc.text, None
        sentence_spans = list(self._tokenizer.span_tokenize(doc.text))
        kept_sentences = []
        original_formatted = []
        last_s = 0
        in_removed_span = False
        for idx, s in enumerate(sentence_spans):
            if idx not in du_lines:
                kept_sentences.append(doc.text[last_s : s[1]])
                if in_removed_span:
                    original_formatted.append("<<<\u001b[0m")
                in_removed_span = False
            elif not in_removed_span:
                in_removed_span = True
                original_formatted.append("\033[91m>>>")
            original_formatted.append(doc.text[last_s : s[1]])
            last_s = s[1]  # use this to include whitespace that is not included in the sentence spans
        if in_removed_span:
            original_formatted.append("<<<\u001b[0m")
        if len(kept_sentences) < len(sentence_spans):
            self.stat_update("removed_sentences", value=len(sentence_spans) - len(kept_sentences))
        self.stat_update("original_sentences", value=len(sentence_spans))
        return "".join(kept_sentences).lstrip(), "".join(original_formatted)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """step method for Filters.
        Drops documents that if .filter() is False

        SentenceDedupFilter reads a DocumentPipeline and removes duplicated sentences found at stage 2
        """
        files = self.data_folder.get_files_shard(rank, world_size, extension=ExtensionHelperSD.stage_2_duplicates)
        assert len(files) == 1, (
            f"n_files / n_tasks should be equal to n_workers, instead {len(files)=}\n{files}.\n"
            f"{world_size=} {rank}"
        )

        du_file = merge_docs(sorted(read_duplicates(files[0])), self.n_sentences)
        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
            for idx, doc in enumerate(data):
                self.stat_update(StatHints.total)
                with self.stats.time_stats:
                    filtered_text, original_formatted = self.remove_dup_sentences(doc, du_lines=du_file.get(idx))
                if (
                    filtered_text == doc.text or len(word_tokenize(filtered_text)) > self.min_doc_words
                ):  # document is kept
                    self.update_doc_stats(doc)
                    if not filtered_text == doc.text and writer:
                        writer.write(dataclasses.replace(doc, text=original_formatted), rank=rank)
                    doc.text = filtered_text
                    yield doc
                elif writer:
                    doc.text = original_formatted
                    writer.write(doc, rank=rank)


class SentenceDedupBuildIndex(PipelineStep):
    type = "🫂 - DEDUP"
    name = "💥 sentence-deduplication build index"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        output_folder: BaseOutputDataFolder,
        index_name: str,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.index_name = index_name

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        assert world_size == 1, "SentenceDedupBuildIndex can only run on a single worker."
        with self.stats.time_stats:
            sig_files = self.data_folder.list_files(extension=ExtensionHelperSD.stage_1_signature)
            sig_readers = [read_sigs(file, file_i) for file_i, file in enumerate(sig_files)]

            pq = [next(sig_reader) for sig_reader in sig_readers]
            heapq.heapify(pq)

            out_f = self.output_folder.open(f"{self.index_name}.{ExtensionHelperSD.index}", mode="wb")

            last = None
            while pq:
                v: HashSig = heapq.heappop(pq)
                if last != v.hash_value:
                    out_f.write(struct.pack("<Q", v.hash_value))
                last = v.hash_value
                new_v = next(sig_readers[v.file_id], None)

                if new_v:
                    heapq.heappush(pq, new_v)
        self.output_folder.close()
