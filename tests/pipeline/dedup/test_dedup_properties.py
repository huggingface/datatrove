import copy
import tempfile
from collections import defaultdict

import pytest


pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

from datatrove.data import Document
from datatrove.pipeline.dedup.exact_dedup import (
    ExactDedupConfig,
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
)
from datatrove.utils.hashing import HashConfig
from tests.utils import require_xxhash


# (precision, hash_fc) tuples: building an xxhash HashConfig imports xxhash, which must not happen at collection
HASH_CONFIGS = [(32, "xxhash"), (64, "xxhash"), (32, "sha1"), (64, "sha1")]


def run_exact_dedup(shards, config, finder_workers):
    with tempfile.TemporaryDirectory() as folder:
        for rank, docs in enumerate(shards):
            signature = ExactDedupSignature(f"{folder}/sigs", config, finder_workers=finder_workers)
            signature.run(iter(copy.deepcopy(docs)), rank, len(shards))
        for rank in range(finder_workers):
            ExactFindDedups(f"{folder}/sigs", f"{folder}/dups", config, save_cluster_size=True).run(
                None, rank, finder_workers
            )
        kept = []
        for rank, docs in enumerate(shards):
            dedup_filter = ExactDedupFilter(f"{folder}/dups", config)
            kept += dedup_filter.run(iter(copy.deepcopy(docs)), rank, len(shards))
        return kept


# Oracle: grouping documents by text in plain Python. Exact dedup keeps exactly one document per distinct text,
# the one with the highest priority, records how many were removed (with save_cluster_size=True), and removes
# nothing when run again. A few short texts make duplicates common. Empty texts are included as input, but how
# they are handled is not asserted here.
@require_xxhash
@given(
    items=st.lists(st.tuples(st.sampled_from(["", "a", "b", "é", "a b", "x" * 50]), st.integers(1, 3)), max_size=20),
    n_shards=st.integers(1, 3),
    finder_workers=st.integers(1, 2),
    hash_config=st.sampled_from(HASH_CONFIGS),
)
def test_exact_dedup_keeps_one_document_per_text(items, n_shards, finder_workers, hash_config):
    docs = [
        Document(text=text, id=str(i), metadata={"priority": priority}) for i, (text, priority) in enumerate(items)
    ]
    config = ExactDedupConfig(
        content_getter=lambda doc: doc.text,
        document_priority=lambda doc: doc.metadata["priority"],
        hash_config=HashConfig(*hash_config),
    )
    kept = run_exact_dedup([docs[i::n_shards] for i in range(n_shards)], config, finder_workers)

    groups = defaultdict(list)
    for doc in docs:
        groups[doc.text].append(doc)
    kept_by_text = defaultdict(list)
    for doc in kept:
        kept_by_text[doc.text].append(doc)
    assert set(kept_by_text) - {""} == set(groups) - {""}
    for text, kept_docs in kept_by_text.items():
        if not text:
            continue
        assert len(kept_docs) == 1
        top_priority = max(doc.metadata["priority"] for doc in groups[text])
        assert kept_docs[0].id in {doc.id for doc in groups[text] if doc.metadata["priority"] == top_priority}
        assert kept_docs[0].metadata.get("duplicate_count", 0) == len(groups[text]) - 1

    kept_again = run_exact_dedup([kept], config, finder_workers)
    assert sorted(doc.id for doc in kept_again) == sorted(doc.id for doc in kept)
