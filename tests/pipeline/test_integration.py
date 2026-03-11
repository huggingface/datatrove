"""Integration tests for higher-order pipeline functionality.

Tests multi-component interactions that aren't covered by unit tests:
- Filter with exclusion_writer writing dropped docs to disk
- Full Reader → Filter → Writer pipeline composition
- JsonlWriter → JsonlReader round-trip with gzip compression
- Writer expand_metadata round-trip
- TokensCounter → LengthCounter chaining
- Stats compute → StatsMerger with multiple stat types
- BaseFilter batched filtering path
"""

import json
import shutil
import tempfile
import unittest
from typing import get_args

from datatrove.data import Document
from datatrove.io import get_datafolder
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.stats import DEFAULT_TOP_K_CONFIG, GROUP, STATS_MERGED_NAME, StatsMerger
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.stats import MetricStatsDict

from ..utils import require_tldextract, require_tokenizers


def make_docs(n: int = 5) -> list[Document]:
    return [
        Document(
            text=f"Document number {i} with some text content.",
            id=str(i),
            metadata={"source": "test", "score": i * 0.1},
        )
        for i in range(n)
    ]


class TestFilterExclusionWriter(unittest.TestCase):
    """Filter with exclusion_writer writes dropped docs to disk with filter_reason."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_dropped_docs_written_with_reason(self):
        exclusion_path = f"{self.tmp_dir}/excluded"
        exclusion_writer = JsonlWriter(exclusion_path, compression=None)
        # Keep only docs with score >= 0.3 (ids 3,4 kept; ids 0,1,2 dropped)
        filt = LambdaFilter(
            filter_function=lambda doc: (True, "ok") if doc.metadata["score"] >= 0.3 else (False, "low_score"),
            exclusion_writer=exclusion_writer,
        )
        docs = make_docs()
        kept = list(filt.run(iter(docs), rank=0, world_size=1))

        assert len(kept) == 2
        assert all(d.metadata["score"] >= 0.3 for d in kept)

        # Read back excluded docs
        reader = JsonlReader(exclusion_path, compression=None)
        excluded = list(reader.run(data=None, rank=0, world_size=1))
        assert len(excluded) == 3
        # filter_reason must be stored in metadata
        for exc_doc in excluded:
            assert exc_doc.metadata["filter_reason"] == "low_score"

    def test_exclusion_writer_stats(self):
        exclusion_path = f"{self.tmp_dir}/excluded"
        exclusion_writer = JsonlWriter(exclusion_path, compression=None)
        filt = LambdaFilter(
            filter_function=lambda doc: doc.metadata["score"] >= 0.3,
            exclusion_writer=exclusion_writer,
        )
        docs = make_docs()
        list(filt.run(iter(docs), rank=0, world_size=1))

        assert filt.stats["total"].total == 5
        assert filt.stats["forwarded"].total == 2
        assert filt.stats["dropped"].total == 3


class TestFullPipelineComposition(unittest.TestCase):
    """Reader → Filter → Writer chain, then read back results."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_reader_filter_writer_roundtrip(self):
        input_path = f"{self.tmp_dir}/input"
        output_path = f"{self.tmp_dir}/output"
        exclusion_path = f"{self.tmp_dir}/excluded"

        # Phase 1: write input data
        with JsonlWriter(input_path, compression=None) as w:
            for doc in make_docs(10):
                w.write(doc)

        # Phase 2: run pipeline — read → filter (keep even ids) → write
        reader = JsonlReader(input_path, compression=None)
        filt = LambdaFilter(
            filter_function=lambda doc: int(doc.id) % 2 == 0,
            exclusion_writer=JsonlWriter(exclusion_path, compression=None),
        )
        writer = JsonlWriter(output_path, compression=None)

        # Chain the pipeline manually like the executor does
        data = reader.run(data=None, rank=0, world_size=1)
        data = filt.run(data, rank=0, world_size=1)
        data = writer.run(data, rank=0, world_size=1)
        # Consume the pipeline
        from collections import deque

        deque(data, maxlen=0)

        # Phase 3: verify outputs
        kept = list(JsonlReader(output_path, compression=None).run(data=None, rank=0, world_size=1))
        excluded = list(JsonlReader(exclusion_path, compression=None).run(data=None, rank=0, world_size=1))

        assert len(kept) == 5
        assert len(excluded) == 5
        assert all(int(d.id) % 2 == 0 for d in kept)
        assert all(int(d.id) % 2 == 1 for d in excluded)
        # Content survives the round-trip
        for doc in kept:
            assert doc.text.startswith("Document number")
            assert "source" in doc.metadata


class TestJsonlGzipRoundTrip(unittest.TestCase):
    """JsonlWriter → JsonlReader with gzip (the default compression)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_gzip_roundtrip(self):
        docs = [
            Document(text="hello world", id="1", metadata={"lang": "en", "score": 0.9}),
            Document(text="foo bar baz", id="2", metadata={"lang": "de", "score": 0.5}),
        ]
        with JsonlWriter(self.tmp_dir, compression="gzip") as w:
            for doc in docs:
                w.write(doc)

        reader = JsonlReader(self.tmp_dir, compression="infer")
        read_docs = list(reader.run(data=None, rank=0, world_size=1))
        assert len(read_docs) == len(docs)
        for read_doc, original in zip(read_docs, docs):
            read_doc.metadata.pop("file_path", None)
            assert read_doc == original

    def test_no_compression_roundtrip(self):
        docs = [
            Document(text="test doc", id="99", metadata={"k": "v"}),
        ]
        with JsonlWriter(self.tmp_dir, compression=None, output_filename="${rank}.jsonl") as w:
            for doc in docs:
                w.write(doc)

        read_docs = list(JsonlReader(self.tmp_dir, compression=None).run(data=None, rank=0, world_size=1))
        assert len(read_docs) == 1
        read_docs[0].metadata.pop("file_path", None)
        assert read_docs[0] == docs[0]


class TestExpandMetadataRoundTrip(unittest.TestCase):
    """Writer with expand_metadata=True flattens metadata; reader reconstructs it."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_expand_metadata_preserves_data(self):
        doc = Document(text="hello", id="1", metadata={"language": "en", "score": 0.95})

        with JsonlWriter(self.tmp_dir, compression=None, expand_metadata=True) as w:
            w.write(doc)

        read_docs = list(JsonlReader(self.tmp_dir, compression=None).run(data=None, rank=0, world_size=1))
        assert len(read_docs) == 1
        rd = read_docs[0]
        # The reader's default adapter puts extra keys into metadata
        assert rd.text == "hello"
        assert rd.id == "1"
        assert rd.metadata["language"] == "en"
        assert rd.metadata["score"] == 0.95


@require_tokenizers
class TestTokensCounterLengthCounterChain(unittest.TestCase):
    """TokensCounter sets metadata['token_count'], LengthCounter reads it for histograms."""

    def test_counter_chain(self):
        from datatrove.pipeline.tokens.counter import LengthCounter, TokensCounter

        docs = [
            Document(text="Hello world", id="1"),
            Document(text="This is a longer sentence with more tokens in it.", id="2"),
            Document(text="Short.", id="3"),
        ]

        counter = TokensCounter(tokenizer_name_or_path="gpt2")
        length_counter = LengthCounter()

        data = counter.run(iter(docs), rank=0, world_size=1)
        data = length_counter.run(data, rank=0, world_size=1)
        results = list(data)

        assert len(results) == 3
        # TokensCounter must have set token_count on every doc
        for doc in results:
            assert "token_count" in doc.metadata
            assert doc.metadata["token_count"] > 0

        # LengthCounter histogram must account for every document
        counts = [doc.metadata["token_count"] for doc in results]
        histogram_total = sum(length_counter.stats[c].total for c in set(counts))
        assert histogram_total == len(results)

        # TokensCounter tracks total tokens
        assert counter.stats["tokens"].total == sum(counts)

    def test_count_eos_token_adds_one(self):
        from datatrove.pipeline.tokens.counter import TokensCounter

        docs = [Document(text="hello", id="1")]

        counter_no_eos = TokensCounter(tokenizer_name_or_path="gpt2", count_eos_token=False)
        counter_eos = TokensCounter(tokenizer_name_or_path="gpt2", count_eos_token=True)

        result_no = list(counter_no_eos.run(iter(docs), rank=0, world_size=1))
        # Need fresh docs since metadata was mutated
        docs2 = [Document(text="hello", id="1")]
        result_eos = list(counter_eos.run(iter(docs2), rank=0, world_size=1))

        assert result_eos[0].metadata["token_count"] == result_no[0].metadata["token_count"] + 1


class DummyMultiStats(BaseStats):
    """Emits multiple stat types per document for testing multi-stat merging."""

    def __init__(self, output_folder, groups=None, histogram_round_digits=2, top_k_config=DEFAULT_TOP_K_CONFIG):
        super().__init__(
            output_folder,
            groups_to_compute=groups or list(get_args(GROUP)),
            histogram_round_digits=histogram_round_digits,
            top_k_config=top_k_config,
        )

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        return {
            "length": len(doc.text),
            "word_count": len(doc.text.split()),
        }


@require_tldextract
class TestStatsComputeAndMerge(unittest.TestCase):
    """Multiple stat types computed across ranks, then merged by StatsMerger."""

    def setUp(self):
        self.tmp_dir = get_datafolder(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_dir.path)

    def test_multi_stat_multi_rank_merge(self):
        docs_rank0 = [
            Document("hello world", "1", metadata={"url": "test1.com"}),
            Document("foo bar", "2", metadata={"url": "test2.com"}),
        ]
        docs_rank1 = [
            Document("another document here", "3", metadata={"url": "test1.com"}),
            Document("short", "4", metadata={"url": "test3.org"}),
        ]

        stats = DummyMultiStats(output_folder=self.tmp_dir, groups=["summary", "histogram"])
        list(stats.run(iter(docs_rank0), rank=0, world_size=2))

        stats2 = DummyMultiStats(output_folder=self.tmp_dir, groups=["summary", "histogram"])
        list(stats2.run(iter(docs_rank1), rank=1, world_size=2))

        merger = StatsMerger(self.tmp_dir, self.tmp_dir)
        list(merger.run(None, rank=0, world_size=1))

        # Verify merged summary stats
        with self.tmp_dir.open(f"summary/length/{STATS_MERGED_NAME}") as f:
            merged = MetricStatsDict.from_dict(json.load(f))
            total_len = sum(len(d.text) for d in docs_rank0 + docs_rank1)
            assert merged["summary"].total == total_len

        with self.tmp_dir.open(f"summary/word_count/{STATS_MERGED_NAME}") as f:
            merged = MetricStatsDict.from_dict(json.load(f))
            total_words = sum(len(d.text.split()) for d in docs_rank0 + docs_rank1)
            assert merged["summary"].total == total_words


class EvenOddBatchFilter(BaseFilter):
    """Drops odd-id docs. Implements filter_batch for batch testing."""

    name = "🧪 EvenOdd Batch"

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        if int(doc.id) % 2 == 0:
            return True
        return False, "odd_id"

    def filter_batch(self, batch: list[Document]) -> list[bool | tuple[bool, str]]:
        return [self.filter(doc) for doc in batch]


class TestBatchedFiltering(unittest.TestCase):
    """BaseFilter with batch_size > 1 uses filter_batch and tracks batch stats."""

    def test_batched_filter_path(self):
        filt = EvenOddBatchFilter(batch_size=3)
        docs = [Document(text=f"doc {i}", id=str(i)) for i in range(10)]
        kept = list(filt.run(iter(docs), rank=0, world_size=1))

        assert len(kept) == 5
        assert all(int(d.id) % 2 == 0 for d in kept)
        assert filt.stats["total"].total == 10
        assert filt.stats["forwarded"].total == 5
        assert filt.stats["dropped"].total == 5
        # batch_size > 1 tracks batch count
        assert filt.stats["batches"].total == 4  # ceil(10/3) = 4

    def test_batched_filter_with_exclusion_writer(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            exclusion_writer = JsonlWriter(tmp_dir, compression=None)
            filt = EvenOddBatchFilter(batch_size=2, exclusion_writer=exclusion_writer)
            docs = [Document(text=f"doc {i}", id=str(i)) for i in range(6)]
            kept = list(filt.run(iter(docs), rank=0, world_size=1))

            assert len(kept) == 3
            excluded = list(JsonlReader(tmp_dir, compression=None).run(data=None, rank=0, world_size=1))
            assert len(excluded) == 3
            for exc_doc in excluded:
                assert exc_doc.metadata["filter_reason"] == "odd_id"
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
