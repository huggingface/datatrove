import json
import shutil
import tempfile
import unittest
from typing import get_args

from datatrove.data import Document
from datatrove.io import get_datafolder
from datatrove.pipeline.stats import (
    DEFAULT_TOP_K_CONFIG,
    GROUP,
    STATS_MERGED_NAME,
    DocStats,
    LangStats,
    LineStats,
    ParagraphStats,
    StatsMerger,
    TokenStats,
    TopKConfig,
    WordsContaminationStats,
    WordStats,
)
from datatrove.pipeline.stats.base import BaseStats
from datatrove.utils.stats import MetricStatsDict
from tests.utils import require_nltk, require_tldextract, require_tokenizers


class DummyStats(BaseStats):
    def __init__(
        self, output_folder, groups=get_args(GROUP), histogram_round_digits=2, top_k_config=DEFAULT_TOP_K_CONFIG
    ):
        super().__init__(
            output_folder,
            groups_to_compute=groups,
            histogram_round_digits=histogram_round_digits,
            top_k_config=top_k_config,
        )

    def extract_stats(self, doc: Document):
        return {"stat": float(doc.text)}


DOCS = [
    Document("1.5", "1", metadata={"url": "test1.co.uk"}),
    Document("2", "1", metadata={"url": "test1.co.uk"}),
    Document("1", "2", metadata={"url": "test2.cz"}),
    Document("1", "3", metadata={"url": "test3.cz"}),
]


@require_tldextract
class TestSummaryStats(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = get_datafolder(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_dir.path)

    def test_grouping(self):
        summary_stats = DummyStats(output_folder=self.tmp_dir)
        list(summary_stats.run(DOCS, 0, 1))

        with self.tmp_dir.open("summary/stat/00000.json") as f:
            stats = MetricStatsDict.from_dict(json.load(f))
            self.assertEqual(stats["summary"].total, 5.5)

        with self.tmp_dir.open("fqdn/stat/00000.json") as f:
            stats = MetricStatsDict.from_dict(json.load(f))
            self.assertEqual(stats["test1.co.uk"].total, 3.5)
            self.assertEqual(stats["test2.cz"].total, 1)
            self.assertEqual(stats["test3.cz"].total, 1)

        with self.tmp_dir.open("suffix/stat/00000.json") as f:
            stats = MetricStatsDict.from_dict(json.load(f))
            self.assertEqual(stats["co.uk"].total, 3.5)
            self.assertEqual(stats["cz"].total, 2)

        with self.tmp_dir.open("histogram/stat/00000.json") as f:
            stats = MetricStatsDict.from_dict(json.load(f))
            self.assertEqual(stats["1.0"].total, 2)
            self.assertEqual(stats["1.5"].total, 1)
            self.assertEqual(stats["2.0"].total, 1)

    def test_histogram_rounding(self):
        summary_stats = DummyStats(output_folder=self.tmp_dir, histogram_round_digits=0)
        list(summary_stats.run(DOCS, 0, 1))

        with self.tmp_dir.open("histogram/stat/00000.json") as f:
            stats = MetricStatsDict.from_dict(json.load(f))
            self.assertEqual(stats["1.0"].total, 2)
            self.assertEqual(stats["2.0"].total, 2)

    def test_compute_top_k(self):
        top_k_config = TopKConfig(top_k=1, top_k_groups=["fqdn"])
        summary_stats = DummyStats(output_folder=self.tmp_dir, top_k_config=top_k_config)
        list(summary_stats.run(DOCS, 0, 1))

        with self.tmp_dir.open("fqdn/stat/00000.json") as f:
            stats = MetricStatsDict.from_dict(json.load(f))
            self.assertEqual(stats["test1.co.uk"].total, 3.5)
            self.assertEqual(stats["test2.cz"].total, 0)

    def test_merging_stats(self):
        summary_stats = DummyStats(output_folder=self.tmp_dir)
        merge_stats = StatsMerger(self.tmp_dir, self.tmp_dir)

        list(summary_stats.run(DOCS[0:2], 0, 2))
        list(summary_stats.run(DOCS[2:4], 1, 2))
        list(merge_stats.run(DOCS, 0, 1))
        with self.tmp_dir.open(f"summary/stat/{STATS_MERGED_NAME}") as f:
            stats = MetricStatsDict.from_dict(json.load(f))
            self.assertEqual(stats["summary"].total, 5.5)

    def test_merging_top_k(self):
        top_k_config = TopKConfig(top_k=1, top_k_groups=["fqdn"])
        summary_stats = DummyStats(output_folder=self.tmp_dir)
        merge_stats = StatsMerger(self.tmp_dir, self.tmp_dir, top_k_config=top_k_config)

        list(summary_stats.run(DOCS[0:2], 0, 2))
        list(summary_stats.run(DOCS[2:4], 1, 2))
        list(merge_stats.run(DOCS, 0, 1))
        with self.tmp_dir.open(f"fqdn/stat/{STATS_MERGED_NAME}") as f:
            stats = MetricStatsDict.from_dict(json.load(f))
            self.assertEqual(stats["test1.co.uk"].total, 3.5)
            self.assertEqual(stats["test2.cz"].total, 0)


@require_tldextract
@require_tokenizers
@require_nltk
class TestStatsModules(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = get_datafolder(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_dir.path)

    def load_computed_means(self, stat_names: list[str]) -> dict:
        def load_stat_total(f) -> dict:
            stat = MetricStatsDict.from_dict(json.load(f))
            return {k: v.total for k, v in stat.items()}

        computed_stats = {}
        for stat in stat_names:
            with self.tmp_dir.open(f"histogram/{stat}/00000.json") as f:
                computed_stats[stat] = load_stat_total(f)
        return computed_stats

    def test_line_stats(self):
        docs = [
            Document("hello\nhow\nhow\nzyou?", "1", metadata={"url": "test.cz"}),
            Document("test test a", "2", metadata={"url": "test.cz"}),
            Document("* Hello", "2", metadata={"url": "test.cz"}),
        ]

        expected_line_stats = {
            "n_lines": {"4": 1, "1": 2},
            "avg_line_length": {"4.0": 1, "11.0": 1, "7.0": 1},
            "short_line_ratio_chars_3": {"0.5": 1, "0.0": 2},
            "long_line_ratio_chars_5": {"0.5": 1, "1.0": 2},
            "bullet_point_lines_ratio": {"0.0": 2, "1.0": 1},
            "line_duplicates": {"0.25": 1, "0.0": 2},
            "line_char_duplicates": {"0.188": 1, "0.0": 2},
        }

        line_stats = LineStats(self.tmp_dir, max_k_chars_per_line_tresholds=[3], min_k_chars_per_line_thresholds=[5])
        list(line_stats.run(docs))

        computed_stats = self.load_computed_means(list(expected_line_stats.keys()))
        self.assertEqual(computed_stats, expected_line_stats)

    def test_doc_stats(self):
        docs = [
            Document("1~", "1", metadata={"url": "test.cz"}),
            Document("Test ...", "2", metadata={"url": "test.cz"}),
        ]
        expected_doc_stats = {
            "length": {"2": 1, "8": 1},
            "white_space_ratio": {"0.125": 1, "0.0": 1},
            "non_alpha_digit_ratio": {"0.5": 2},
            "digit_ratio": {"0.5": 1, "0.0": 1},
            "uppercase_ratio": {"0.125": 1, "0.0": 1},
            "elipsis_ratio": {"0.375": 1, "0.0": 1},
            "punctuation_ratio": {"0.5": 1, "0.375": 1},
        }
        doc_stats = DocStats(self.tmp_dir)

        list(doc_stats.run(docs))
        computed_stats = self.load_computed_means(list(expected_doc_stats.keys()))
        self.assertEqual(computed_stats, expected_doc_stats)

    def test_word_stats(self):
        docs = [
            Document("okay\nokay", "1", metadata={"url": "test.cz"}),
            Document("test test of", "2", metadata={"url": "test.cz"}),
            Document("Looooooooong", "3", metadata={"url": "test.cz"}),
        ]

        expected_word_stats = {
            "n_words": {"2": 1, "3": 1, "1": 1},
            "avg_word_length": {"3.333": 1, "4.0": 1, "12.0": 1},
            "avg_words_per_line": {"1.0": 2, "3.0": 1},
            "short_word_ratio_3": {"0.333": 1, "0.0": 2},
            "long_word_ratio_7": {"0.0": 2, "1.0": 1},
            "type_token_ratio": {"0.5": 1, "0.667": 1, "1.0": 1},
            "uppercase_word_ratio": {"0.0": 3},
            "capitalized_word_ratio": {"0.0": 2, "1.0": 1},
            "stop_word_ratio": {"0.0": 2, "0.333": 1},
        }
        word_stats = WordStats(
            self.tmp_dir,
            short_word_max_chars_threshold=[3],
            long_word_max_chars_threshold=[7],
            groups_to_compute=["histogram"],
        )
        list(word_stats.run(docs))
        computed_stats = self.load_computed_means(list(expected_word_stats.keys()))
        self.assertEqual(computed_stats, expected_word_stats)

    def test_words_contamination(self):
        docs = [
            Document("chat gpt loves the word delve and delve is word", "1", metadata={"url": "test.cz"}),
            Document("chat gpt doesn't prefer any words", "2", metadata={"url": "test.cz"}),
        ]

        expected_words_contamination = {
            "words_contamination_delve": {"0.2": 1, "0.0": 1},
        }

        contamination_stats = WordsContaminationStats(self.tmp_dir, ["delve"])
        list(contamination_stats.run(docs))

        computed_stats = self.load_computed_means(list(expected_words_contamination.keys()))
        self.assertEqual(computed_stats, expected_words_contamination)

    def test_token_counter(self):
        docs = [
            Document("hi how are you ?", "1", metadata={"url": "test.cz"}),
            Document(" hi hi", "2", metadata={"url": "test.cz"}),
        ]

        expected_token_counter = {
            "token_count": {"5": 1, "2": 1},
        }

        token_counter = TokenStats(self.tmp_dir)
        list(token_counter.run(docs))

        computed_stats = self.load_computed_means(list(expected_token_counter.keys()))
        self.assertEqual(computed_stats, expected_token_counter)

    def test_lang_stats(self):
        docs = [
            Document("This is pure english text", "1", metadata={"url": "test.cz"}),
            Document("Toto je český text", "2", metadata={"url": "test.cz"}),
        ]

        expected_lang_stats = {
            "fasttext_en": {"0.887": 1, "0.0": 1},
        }

        lang_stats = LangStats(self.tmp_dir, language="en")
        list(lang_stats.run(docs))

        computed_stats = self.load_computed_means(list(expected_lang_stats.keys()))
        self.assertEqual(computed_stats, expected_lang_stats)

    def test_paragraph_stats(self):
        docs = [
            Document(
                "paragraph one\n\nparagraph two\n\nshort\n\nvery very long one", "1", metadata={"url": "test.cz"}
            ),
        ]

        expected_paragraph_stats = {
            "n_paragraphs": {"4": 1},
            "avg_paragraph_length": {"12.25": 1},
            "short_paragraph_ratio_5": {"0.25": 1},
            "long_paragraph_ratio_15": {"0.25": 1},
        }
        paragraph_stats = ParagraphStats(
            self.tmp_dir, short_paragraph_max_chars_threshold=[5], long_paragraph_max_chars_threshold=[15]
        )
        list(paragraph_stats.run(docs))

        computed_stats = self.load_computed_means(list(expected_paragraph_stats.keys()))
        self.assertEqual(computed_stats, expected_paragraph_stats)
