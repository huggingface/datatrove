import copy
import random
import shutil
import string
import tempfile
import unittest

from datatrove.data import Document
from datatrove.pipeline.dedup.url_dedup import (
    UrlDedupConfig,
    UrlDedupFilter,
    UrlDedupSignature,
    UrlFindDedups,
)

from ..utils import require_nltk


def get_random_string(n: int = 20):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n)) + "."


DOCS = [
    Document(text="", metadata={"url": "https://example.com"}, id="1"),
    Document(text="", metadata={"url": "https://example.com"}, id="2"),
    Document(text="", metadata={"url": "https://new-site.com"}, id="3"),
    Document(text="", metadata={"url": "https://example.com"}, id="4"),
    Document(text="", metadata={"url": "https://example2.com"}, id="5"),
]

DOCS_1 = DOCS[:2]
DOCS_2 = DOCS[2:]


@require_nltk
class SentenceDedup(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_url_deduplication(self):
        signature_creation = UrlDedupSignature(output_folder=self.tmp_dir + "/sigs")
        find_duplicates = UrlFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            lines_to_buffer=1000,
        )
        dedup_filter = UrlDedupFilter(data_folder=self.tmp_dir + "/dups")

        signature_creation(data=DOCS)
        find_duplicates()
        docs = list(dedup_filter(data=copy.deepcopy(DOCS)))
        self.assertEqual(len(docs), 3)
        self.assertEqual(
            set(doc.metadata["url"] for doc in docs),
            set([doc.metadata["url"] for doc in DOCS]),
        )

    def test_url_deduplication_with_priority_highest_id(self):
        config = UrlDedupConfig(document_priority=lambda x: int(x.id))

        signature_creation = UrlDedupSignature(
            output_folder=self.tmp_dir + "/sigs", config=config
        )
        find_duplicates = UrlFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
        )
        dedup_filter = UrlDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS)
        find_duplicates()
        docs = list(dedup_filter(data=copy.deepcopy(DOCS)))

        expected_ids = [3, 4, 5]
        self.assertEqual(len(docs), 3)
        self.assertEqual(set(int(doc.id) for doc in docs), set(expected_ids))

    def test_url_deduplication_with_priority_lowest_id(self):
        config = UrlDedupConfig(document_priority=lambda x: 5 - int(x.id))

        signature_creation = UrlDedupSignature(
            output_folder=self.tmp_dir + "/sigs", config=config
        )
        find_duplicates = UrlFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
        )
        dedup_filter = UrlDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS)
        find_duplicates()
        docs = list(dedup_filter(data=copy.deepcopy(DOCS)))

        expected_ids = [1, 3, 5]
        self.assertEqual(len(docs), 3)
        self.assertEqual(set(int(doc.id) for doc in docs), set(expected_ids))

    def test_url_deduplication_with_normalization(self):
        config = UrlDedupConfig(url_normalizer=lambda x: x.replace("2", ""))

        signature_creation = UrlDedupSignature(
            output_folder=self.tmp_dir + "/sigs", config=config
        )
        find_duplicates = UrlFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
        )
        dedup_filter = UrlDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS)
        find_duplicates()
        docs = list(dedup_filter(data=copy.deepcopy(DOCS)))

        self.assertEqual(len(docs), 2)
        self.assertEqual(
            set(doc.metadata["url"] for doc in docs),
            set(["https://example.com", "https://new-site.com"]),
        )

    def test_sd_worker(self):
        config = UrlDedupConfig(document_priority=lambda x: int(x.id))
        signature_creation = UrlDedupSignature(
            output_folder=self.tmp_dir + "/sigs", config=config
        )

        find_duplicates = UrlFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
        )
        dedup_filter = UrlDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS_1, rank=0, world_size=2)
        signature_creation(data=DOCS_2, rank=1, world_size=2)
        find_duplicates()

        dedup_1 = list(dedup_filter(data=copy.deepcopy(DOCS_1), rank=0, world_size=2))
        dedup_2 = list(dedup_filter(data=copy.deepcopy(DOCS_2), rank=1, world_size=2))

        self.assertEqual(len(dedup_1), 0)
        self.assertEqual(len(dedup_2), 3)
        self.assertEqual(
            set(doc.metadata["url"] for doc in dedup_2),
            set(doc.metadata["url"] for doc in DOCS),
        )

    def test_distributed_find_dups(self):
        config = UrlDedupConfig(document_priority=lambda x: int(x.id))

        signature_creation = UrlDedupSignature(
            output_folder=self.tmp_dir + "/sigs", finder_workers=50, config=config
        )

        find_duplicates = UrlFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
        )
        dedup_filter = UrlDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS_1, rank=0, world_size=2)
        signature_creation(data=DOCS_2, rank=1, world_size=2)
        for rank in range(50):
            find_duplicates(rank=rank, world_size=50)

        dedup_docs = list(
            dedup_filter(data=copy.deepcopy(DOCS_1), rank=0, world_size=2)
        )

        dedup_docs_2 = list(
            dedup_filter(data=copy.deepcopy(DOCS_2), rank=1, world_size=2)
        )
        self.assertEqual(len(dedup_docs), 0)
        self.assertEqual(len(dedup_docs_2), 3)
        self.assertEqual(
            set(doc.metadata["url"] for doc in dedup_docs_2),
            set(doc.metadata["url"] for doc in DOCS),
        )
