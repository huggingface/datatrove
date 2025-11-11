import copy
import shutil
import tempfile
import unittest

from datatrove.data import Document
from datatrove.pipeline.dedup.exact_dedup import (
    ExactDedupBuildIndex,
    ExactDedupConfig,
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
)
from tests.utils import require_xxhash, use_hash_configs


DOCS = [
    Document(text="", metadata={"url": "https://example.com"}, id="1"),
    Document(text="", metadata={"url": "https://example.com"}, id="2"),
    Document(text="", metadata={"url": "https://new-site.com"}, id="3"),
    Document(text="", metadata={"url": "https://example.com"}, id="4"),
    Document(text="", metadata={"url": "https://example2.com"}, id="5"),
]

INDEX = [
    Document(text="", metadata={"url": "https://example.com"}, id="1"),
    Document(text="", metadata={"url": "https://example2.com"}, id="2"),
]

DOCS_1 = DOCS[:2]
DOCS_2 = DOCS[2:]


@require_xxhash
class UrlDedup(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_url_deduplication(self):
        config = ExactDedupConfig(content_getter=lambda doc: doc.metadata.get("url", ""))
        signature_creation = ExactDedupSignature(output_folder=self.tmp_dir + "/sigs", config=config)
        find_duplicates = ExactFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
            lines_to_buffer=1000,
        )
        dedup_filter = ExactDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS)
        find_duplicates()
        docs = list(dedup_filter(data=copy.deepcopy(DOCS)))
        self.assertEqual(len(docs), 3)
        self.assertEqual(
            {doc.metadata["url"] for doc in docs},
            {doc.metadata["url"] for doc in DOCS},
        )

    def test_url_deduplication_with_priority_highest_id(self):
        config = ExactDedupConfig(
            content_getter=lambda doc: doc.metadata.get("url", ""), document_priority=lambda x: int(x.id)
        )

        signature_creation = ExactDedupSignature(output_folder=self.tmp_dir + "/sigs", config=config)
        find_duplicates = ExactFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
        )
        dedup_filter = ExactDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS)
        find_duplicates()
        docs = list(dedup_filter(data=copy.deepcopy(DOCS)))

        expected_ids = [3, 4, 5]
        self.assertEqual(len(docs), 3)
        self.assertEqual({int(doc.id) for doc in docs}, set(expected_ids))

    def test_url_deduplication_with_priority_lowest_id(self):
        config = ExactDedupConfig(
            content_getter=lambda doc: doc.metadata.get("url", ""), document_priority=lambda x: 5 - int(x.id) + 1
        )

        signature_creation = ExactDedupSignature(output_folder=self.tmp_dir + "/sigs", config=config)
        find_duplicates = ExactFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
        )
        dedup_filter = ExactDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS)
        find_duplicates()
        docs = list(dedup_filter(data=copy.deepcopy(DOCS)))

        expected_ids = [1, 3, 5]
        self.assertEqual(len(docs), 3)
        self.assertEqual({int(doc.id) for doc in docs}, set(expected_ids))

    def test_url_deduplication_with_normalization(self):
        config = ExactDedupConfig(content_getter=lambda doc: doc.metadata.get("url", "").replace("2", ""))

        signature_creation = ExactDedupSignature(output_folder=self.tmp_dir + "/sigs", config=config)
        find_duplicates = ExactFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
        )
        dedup_filter = ExactDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS)
        find_duplicates()
        docs = list(dedup_filter(data=copy.deepcopy(DOCS)))

        self.assertEqual(len(docs), 2)
        self.assertEqual(
            {doc.metadata["url"] for doc in docs},
            {"https://example.com", "https://new-site.com"},
        )

    def test_url_deduplication_with_index(self):
        config = ExactDedupConfig(content_getter=lambda doc: doc.metadata.get("url", ""))
        signature_creation = ExactDedupSignature(output_folder=self.tmp_dir + "/sigs", config=config)
        index_signature_creation = ExactDedupSignature(output_folder=self.tmp_dir + "/index_sigs", config=config)
        build_index = ExactDedupBuildIndex(
            data_folder=self.tmp_dir + "/index_sigs",
            output_folder=self.tmp_dir + "/index",
            index_name="index",
            config=config,
            lines_to_buffer=1000,
        )
        find_duplicates = ExactFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            index_folder=self.tmp_dir + "/index",
            output_folder=self.tmp_dir + "/dups",
            config=config,
            lines_to_buffer=1000,
        )
        dedup_filter = ExactDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        index_signature_creation(data=INDEX)
        build_index()
        signature_creation(data=DOCS)
        find_duplicates()
        docs = list(dedup_filter(data=copy.deepcopy(DOCS)))
        self.assertEqual(len(docs), 1)
        self.assertEqual(
            {doc.metadata["url"] for doc in docs},
            {doc.metadata["url"] for doc in DOCS} - {doc.metadata["url"] for doc in INDEX},
        )

    def test_sd_worker(self):
        config = ExactDedupConfig(
            content_getter=lambda doc: doc.metadata.get("url", ""), document_priority=lambda x: int(x.id)
        )
        signature_creation = ExactDedupSignature(output_folder=self.tmp_dir + "/sigs", config=config)

        find_duplicates = ExactFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
        )
        dedup_filter = ExactDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS_1, rank=0, world_size=2)
        signature_creation(data=DOCS_2, rank=1, world_size=2)
        find_duplicates()

        dedup_1 = list(dedup_filter(data=copy.deepcopy(DOCS_1), rank=0, world_size=2))
        dedup_2 = list(dedup_filter(data=copy.deepcopy(DOCS_2), rank=1, world_size=2))

        self.assertEqual(len(dedup_1), 0)
        self.assertEqual(len(dedup_2), 3)
        self.assertEqual(
            {doc.metadata["url"] for doc in dedup_2},
            {doc.metadata["url"] for doc in DOCS},
        )

    @use_hash_configs()
    def test_distributed_find_dups(self, hash_config):
        config = ExactDedupConfig(
            content_getter=lambda doc: doc.metadata.get("url", ""),
            document_priority=lambda x: int(x.id),
            hash_config=hash_config,
        )

        signature_creation = ExactDedupSignature(
            output_folder=self.tmp_dir + "/sigs", finder_workers=50, config=config
        )

        find_duplicates = ExactFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
        )
        dedup_filter = ExactDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS_1, rank=0, world_size=2)
        signature_creation(data=DOCS_2, rank=1, world_size=2)
        for rank in range(50):
            find_duplicates(rank=rank, world_size=50)

        dedup_docs = list(dedup_filter(data=copy.deepcopy(DOCS_1), rank=0, world_size=2))

        dedup_docs_2 = list(dedup_filter(data=copy.deepcopy(DOCS_2), rank=1, world_size=2))
        self.assertEqual(len(dedup_docs), 0)
        self.assertEqual(len(dedup_docs_2), 3)
        self.assertEqual(
            {doc.metadata["url"] for doc in dedup_docs_2},
            {doc.metadata["url"] for doc in DOCS},
        )

    def test_cluster_size(self):
        """Test that duplicate_count metadata is correctly added to kept documents"""
        config = ExactDedupConfig(
            content_getter=lambda doc: doc.metadata.get("url", ""), document_priority=lambda x: int(x.id)
        )

        signature_creation = ExactDedupSignature(output_folder=self.tmp_dir + "/sigs", config=config)
        find_duplicates = ExactFindDedups(
            data_folder=self.tmp_dir + "/sigs",
            output_folder=self.tmp_dir + "/dups",
            config=config,
            save_cluster_size=True,
        )
        dedup_filter = ExactDedupFilter(data_folder=self.tmp_dir + "/dups", config=config)

        signature_creation(data=DOCS)
        find_duplicates()
        docs = list(dedup_filter(data=copy.deepcopy(DOCS)))

        # Should keep 3 documents: one from each unique URL cluster
        self.assertEqual(len(docs), 3)

        # Create a mapping of doc_id to duplicate_count
        doc_counts = {doc.id: doc.metadata.get("duplicate_count", 0) for doc in docs}

        # Doc 4 should be kept (highest priority for "https://example.com" cluster)
        # and should have duplicate_count=2 (docs 1 and 2 are duplicates)
        self.assertIn("4", doc_counts)
        self.assertEqual(doc_counts["4"], 2)

        # Doc 3 should be kept (only one with "https://new-site.com")
        # and should have duplicate_count=0 (no duplicates)
        self.assertIn("3", doc_counts)
        self.assertEqual(doc_counts["3"], 0)

        # Doc 5 should be kept (only one with "https://example2.com")
        # and should have duplicate_count=0 (no duplicates)
        self.assertIn("5", doc_counts)
        self.assertEqual(doc_counts["5"], 0)
