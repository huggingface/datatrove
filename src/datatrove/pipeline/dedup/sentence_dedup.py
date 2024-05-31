"""'To deduplicate the data set, we discarded all but one of any three-sentence span
occurring more than once in the data set.'

from: https://jmlr.org/papers/volume21/20-074/20-074.pdf (C4)

# get hashes for each doc and write them down

"""

import contextlib
import dataclasses
import heapq
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Generator

import numpy as np
from fsspec.spec import AbstractBufferedFile
from tqdm import tqdm

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.binaryio import read_np_from_file, read_tuples_from_file
from datatrove.utils.hashing import HashConfig, create_hash_func
from datatrove.utils.logging import logger
from datatrove.utils.text import SPLIT_TEXT_SENTENCES, TextNormConfig, ngrams, simplify_text, split_into_parts
from datatrove.utils.typeshelper import ExtensionHelperSD, Languages, StatHints

from ...utils.word_tokenizers import load_word_tokenizer
from ..writers.disk_base import DiskWriter


@dataclass
class SentDedupConfig:
    n_sentences: int = 3
    split_sentences: bool = True  # set to False to split on \n instead
    only_dedup_in_index: bool = True
    min_doc_words: int = 50
    min_num_sentences: int = 3  # remove docs that end up with fewer than 3 sentences
    min_words_to_remove_span: int = 0
    norm_config: TextNormConfig = field(default_factory=TextNormConfig)
    hash_config: HashConfig = field(default_factory=HashConfig)


@dataclass(order=True)
class HashSig:
    # this also determines the sorting order
    # hash_value needs to come first as that's what we match on
    # file_id should come after doc_id so that hashes from the index (sent_id=doc_id=-1) come up first
    hash_value: int
    doc_id: int
    file_id: int = None
    sent_id: int = None
    file_stem: str = None

    def is_from_index(self):
        return self.doc_id == self.sent_id == -1


class SentenceDedupSignature(PipelineStep):
    """SentenceDedup: First pipeline step

        Creates a signature for each sentence in each document. Each HashSig has n hash, the doc id and the sentence idx. Before saving
        them the hashes are sorted.

    Args:
        output_folder: folder where signatures are saved
    """

    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 1"

    def __init__(
        self,
        output_folder: DataFolderLike,
        finder_workers: int = 1,
        config: SentDedupConfig = None,
        language: str = Languages.english,
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        if finder_workers <= 0:
            raise ValueError("finder_workers must be >= 1")
        elif finder_workers > 1:
            logger.warning(f"Remember to also set the name of tasks of the finder block to {finder_workers=}!")
        self.finder_workers = finder_workers
        self.config = config or SentDedupConfig()
        self.hash_fc = create_hash_func(config.hash_config)
        self.language = language
        self.tokenizer = load_word_tokenizer(language)

    def save_hashes(self, rank: int, signatures):
        # explicitly define little endianness
        signatures = np.array(
            signatures, dtype=[("hash", self.config.hash_config.np_descr), ("doc", "<u4"), ("sent", "<u2")]
        )
        signatures.sort(axis=0)

        hashes_per_worker = self.config.hash_config.max // self.finder_workers
        left_idx = 0
        for hash_i in range(self.finder_workers):
            with self.output_folder.open(
                f"{hash_i:04d}/{rank:05d}{ExtensionHelperSD.stage_1_signature}", mode="wb"
            ) as f:
                # last bucket needs to have everything
                right_hash = (
                    (hash_i + 1) * hashes_per_worker
                    if hash_i != self.finder_workers - 1
                    else self.config.hash_config.max
                )
                # find last hash that goes in this bucket. This obeys the following rule:
                # signatures['hash'][right_idx - 1] <= right_hash <= signatures['hash'][right_idx]
                right_idx = left_idx + signatures["hash"][left_idx:].searchsorted(right_hash, side="right")
                # save to file
                if right_idx > left_idx:
                    if self.output_folder.is_local():
                        signatures[left_idx:right_idx].tofile(f)
                    else:
                        f.write(signatures[left_idx:right_idx].tobytes())
                left_idx = right_idx
                # we've reached the end of our data
                if right_idx >= len(signatures):
                    break

    def get_hashes(self, doc: Document, doc_idx: int) -> list[None] | list[tuple[int, int, int]]:
        sentences = self.tokenizer.sent_tokenize(doc.text) if self.config.split_sentences else doc.text.splitlines()
        if len(sentences) < self.config.n_sentences:
            return []

        sentences_tokens = [simplify_text(sent, self.config.norm_config) for sent in sentences]
        n_sent_grams: list = [" ".join(x) for x in ngrams(sentences_tokens, self.config.n_sentences)]
        hashes = [
            (self.hash_fc(n_sent_gram), doc_idx, sentence_idx)
            for sentence_idx, n_sent_gram in enumerate(n_sent_grams)
            if n_sent_gram.strip() != ""  # we actually do not want to remove all the \n everywhere
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


def read_sigs(
    file: AbstractBufferedFile,
    file_id: int,
    config: SentDedupConfig,
    index_file: bool = False,
    lines_to_buffer: int = 5,
) -> Generator[HashSig, None, None]:
    line_format = f"{config.hash_config.struct_format}IH" if not index_file else config.hash_config.struct_format
    file_stem = Path(file.path).name.removesuffix(ExtensionHelperSD.stage_1_signature)
    last = None
    with file as f:
        for data in read_tuples_from_file(f, line_format, lines_to_buffer=lines_to_buffer):
            assert last is None or data[0] >= last, f"Hash order error. {f.tell()=}, {data[0]=}, {last=}"
            last = data[0]
            yield (
                HashSig(hash_value=data[0], doc_id=-1, file_id=file_id, sent_id=-1, file_stem=file_stem)
                if index_file
                else HashSig(file_id=file_id, hash_value=data[0], doc_id=data[1], sent_id=data[2], file_stem=file_stem)
            )


class SentenceFindDedups(PipelineStep):
    """SentenceDedup: Second pipeline step

        SentenceFindDedups runs on a single worker. It reads all the signatures from the previous step and loads them
        in a priority queue to check for duplicates. If a duplicate is found its document id and sentence id are saved.

    Args:
        data_folder: data folder where signatures are saved
        output_folder: folder where duplicates are saved
        index_folder: folder where index files are saved
        only_dedup_in_index: only dedup in index
    """

    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 2"

    def __init__(
        self,
        data_folder: DataFolderLike,
        output_folder: DataFolderLike,
        index_folder: DataFolderLike = None,
        config: SentDedupConfig = None,
        lines_to_buffer: int = 5,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.output_folder = get_datafolder(output_folder)
        self.index_folder = get_datafolder(index_folder) if index_folder else None
        self.config = config or SentDedupConfig()
        self.lines_to_buffer = lines_to_buffer

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        with self.stats.time_stats:
            if world_size == 1:
                # check that there was not a mistake in setting this values
                sig_files = self.data_folder.list_files(glob_pattern="*/*" + ExtensionHelperSD.stage_1_signature)
                if any(not sig_file.startswith("0000/") for sig_file in sig_files):
                    raise ValueError(
                        f"{world_size=} but found sig files for different hash buckets. Set tasks=finder_workers"
                    )
            else:
                sig_files = self.data_folder.list_files(
                    subdirectory=f"{rank:04d}", glob_pattern=ExtensionHelperSD.stage_1_signature
                )
            sig_readers = [
                read_sigs(file, file_i, config=self.config, lines_to_buffer=self.lines_to_buffer)
                for file_i, file in enumerate(self.data_folder.open_files(sig_files))
            ]
            index_files = self.index_folder.list_files() if self.index_folder else None
            if index_files:
                logger.info(f"Found index file(s): {', '.join(index_files)}")
                sig_readers.extend(
                    [
                        read_sigs(
                            file,
                            len(sig_readers) + file_i,
                            config=self.config,
                            index_file=True,
                            lines_to_buffer=self.lines_to_buffer,
                        )
                        for file_i, file in enumerate(self.data_folder.open_files(index_files))
                    ]
                )

            logger.info(f"Initializing pq with {len(sig_readers)} files.")
            with ThreadPoolExecutor() as executor:
                pq = [
                    x
                    for x in tqdm(
                        executor.map(lambda x: next(x, None), sig_readers),
                        total=len(sig_readers),
                        desc="Initializing pq...",
                    )
                    if x
                ]
            heapq.heapify(pq)
            logger.info("PQ initialized.")

            output_mg = self.output_folder.get_output_file_manager(mode="wb")

            packer = struct.Struct("<IH")

            last: HashSig | None = None
            while pq:
                v: HashSig = heapq.heappop(pq)
                if (
                    last and last.hash_value == v.hash_value and not v.is_from_index()
                ):  # we never want to match samples from the index itself
                    out_filename = f"{rank:04d}/{v.file_stem}{ExtensionHelperSD.stage_2_duplicates}"
                    # the previous one we are matching against is part of the index
                    # OR there are no index files
                    # OR we are also matching within the main dataset
                    if last.is_from_index() or not index_files or not self.config.only_dedup_in_index:
                        output_mg.write(out_filename, packer.pack(v.doc_id, v.sent_id))
                last = v
                new_v = next(sig_readers[v.file_id], None)

                if new_v:
                    heapq.heappush(pq, new_v)

        output_mg.close()


class SentenceDedupFilter(PipelineStep):
    """SentenceDedup: Third pipeline step

        SentenceDedupFilter reads a DocumentPipeline and removes duplicated sentences found at stage 2

    Args:
        data_folder: data folder to get duplicate files.
        n_sentences: n_sentences where duplicates are checked. Should match step1
        min_doc_words: min amount of words (after removing duplicate sentences) to keep a document
        exclusion_writer: writer to save excluded documents
    """

    type = "🫂 - DEDUPS"
    name = "💥 sentence-deduplication stage 3"

    def __init__(
        self,
        data_folder: DataFolderLike,
        config: SentDedupConfig = None,
        exclusion_writer: DiskWriter = None,
        language: str = Languages.english,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.config = config or SentDedupConfig()
        self.tokenizer = load_word_tokenizer(language)
        self.exclusion_writer = exclusion_writer
        self.language = language

    def read_duplicates(self, file: BinaryIO) -> np.ndarray:
        """Helper function to read duplicates from a binary file storing (doc_id, sent_id) pairs as created by the second stage."""
        return read_np_from_file(
            file, dtype=np.dtype([("doc", "<u4"), ("sent", "<u2")]), is_local_file=self.data_folder.is_local()
        )

    def remove_dup_sentences(self, doc: Document, du_lines: np.ndarray) -> tuple[str, str]:
        sentence_spans = (
            list(self.tokenizer.span_tokenize(doc.text)) if self.config.split_sentences else doc.text.splitlines()
        )
        kept_sentences = []
        original_formatted = []
        last_s = 0
        du_line_idx = 0  # pointer for duplicate lines
        drop_until = 0  # used to keep track of last matched span's end
        removed_span = []
        for idx, s in enumerate(sentence_spans):
            line_text = doc.text[last_s : s[1]] if self.config.split_sentences else s
            # track / increment dup_line ref
            if du_line_idx < len(du_lines):
                if du_lines[du_line_idx] < idx:
                    raise ValueError("Error with duplicate line index")
                elif du_lines[du_line_idx] == idx:
                    drop_until = idx + self.config.n_sentences
                    du_line_idx += 1

            # if outside the range, we keep this line/sent
            if idx >= drop_until:
                if removed_span:
                    original_formatted.append("<<<")
                    if (
                        self.config.min_words_to_remove_span > 0
                        and len(self.tokenizer.word_tokenize("\n".join(removed_span)))
                        < self.config.min_words_to_remove_span
                    ):
                        kept_sentences.extend(removed_span)
                    removed_span.clear()
                kept_sentences.append(line_text)
            elif not removed_span:
                removed_span.append(line_text)
                original_formatted.append(">>>")
            original_formatted.append(line_text)
            if self.config.split_sentences:
                last_s = s[1]  # use this to include whitespace that is not included in the sentence spans
        if removed_span:
            original_formatted.append("<<<")
            if (
                self.config.min_words_to_remove_span > 0
                and len(self.tokenizer.word_tokenize("\n".join(removed_span))) < self.config.min_words_to_remove_span
            ):
                kept_sentences.extend(removed_span)
        if len(kept_sentences) < len(sentence_spans):
            self.stat_update("removed_sentences", value=len(sentence_spans) - len(kept_sentences))
        self.stat_update("original_sentences", value=len(sentence_spans))
        merge_char = "" if self.config.split_sentences else "\n"
        return merge_char.join(kept_sentences).lstrip(), merge_char.join(original_formatted)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """step method for Filters.
        Drops documents that if .filter() is False

        SentenceDedupFilter reads a DocumentPipeline and removes duplicated sentences found at stage 2
        """
        folders = self.data_folder.list_files(include_directories=True, recursive=False)
        # for performance reasons when having for instance 12k*10k files
        files = [
            f
            for f in [f"{folder}/{rank:05d}{ExtensionHelperSD.stage_2_duplicates}" for folder in folders]
            if self.data_folder.exists(f)
        ]

        logger.info(f"Loading duplicate indexes from {len(files)} results files.")

        all_dups = np.array([], dtype=[("doc", "<u4"), ("sent", "<u2")])
        if files:
            with ThreadPoolExecutor() as pool:
                all_dups = np.concatenate(
                    list(tqdm(pool.map(self.read_duplicates, self.data_folder.open_files(files)), total=len(files))),
                    axis=0,
                )
            all_dups.sort()
        _, doc_starts = np.unique(all_dups["doc"], return_index=True)

        logger.info("Loaded duplicate indexes.")

        dups_doc_i = 0
        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
            for doc_idx, doc in enumerate(data):
                self.stat_update(StatHints.total)
                with self.stats.time_stats:
                    if dups_doc_i >= len(doc_starts) or all_dups["doc"][doc_starts[dups_doc_i]] > doc_idx:
                        filtered_text, original_formatted = doc.text, None
                    else:
                        sents_span_l, sents_span_r = (
                            doc_starts[dups_doc_i],
                            doc_starts[dups_doc_i + 1] if dups_doc_i + 1 < len(doc_starts) else None,
                        )
                        filtered_text, original_formatted = self.remove_dup_sentences(
                            doc, all_dups["sent"][sents_span_l:sents_span_r]
                        )
                        dups_doc_i += 1

                if (
                    (
                        filtered_text == doc.text  # no change
                        or (
                            (
                                # min doc words
                                self.config.min_doc_words <= 0
                                or len(self.tokenizer.word_tokenize(filtered_text)) >= self.config.min_doc_words
                            )
                            and (
                                # min num sentences
                                self.config.min_num_sentences <= 0
                                or len(split_into_parts(filtered_text, SPLIT_TEXT_SENTENCES, self.language))
                                >= self.config.min_num_sentences
                            )
                        )
                    )
                    and filtered_text  # can not be completely empty
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
    """SentenceDedup: Only build an index

    Args:
        data_folder: data folder to get signature files.
        output_folder: folder where index is saved
        index_name: name of the index
    """

    type = "🫂 - DEDUP"
    name = "💥 sentence-deduplication build index"

    def __init__(
        self,
        data_folder: DataFolderLike,
        output_folder: DataFolderLike,
        index_name: str,
        config: SentDedupConfig = None,
        lines_to_buffer: int = 5,
    ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        self.output_folder = get_datafolder(output_folder)
        self.index_name = index_name
        self.lines_to_buffer = lines_to_buffer
        self.config = config or SentDedupConfig()

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        assert world_size == 1, "SentenceDedupBuildIndex can only run on a single worker."
        with self.stats.time_stats:
            sig_files = self.data_folder.list_files(glob_pattern=ExtensionHelperSD.stage_1_signature)
            sig_readers = [
                read_sigs(file, file_i, self.config, lines_to_buffer=self.lines_to_buffer)
                for file_i, file in enumerate(self.data_folder.open_files(sig_files))
            ]

            pq = [next(sig_reader) for sig_reader in sig_readers]
            heapq.heapify(pq)

            with self.output_folder.open(f"{self.index_name}.{ExtensionHelperSD.index}", mode="wb") as out_f:
                last = None
                while pq:
                    v: HashSig = heapq.heappop(pq)
                    if last != v.hash_value:
                        out_f.write(struct.pack(f"<{self.config.hash_config.struct_format}", v.hash_value))
                    last = v.hash_value
                    new_v = next(sig_readers[v.file_id], None)

                    if new_v:
                        heapq.heappush(pq, new_v)
