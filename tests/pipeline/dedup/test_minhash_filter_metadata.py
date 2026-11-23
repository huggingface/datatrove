import struct
from pathlib import Path

import pytest

from datatrove.data import Document
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupCluster, MinhashDedupFilter


pytest.importorskip("xxhash")


@pytest.mark.parametrize("load_ids, load_sizes", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("empty_remove_file", [False, True])
@pytest.mark.parametrize("lines_to_buffer", [1, 5])
def test_cluster_metadata_on_rank_without_removals(
    tmp_path: Path, load_ids: bool, load_sizes: bool, empty_remove_file: bool, lines_to_buffer: int
) -> None:
    """Load metadata for cross-rank cluster representatives without requiring removals."""
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    (pairs / "00000_00.dups").write_bytes(struct.pack("<8I", 0, 1, 1, 0, 0, 3, 1, 2))
    clusters = tmp_path / "clusters"
    MinhashDedupCluster(
        str(pairs),
        str(clusters),
        config=MinhashConfig(num_buckets=1),
        save_cluster_id=True,
        save_cluster_size=True,
    ).run()
    assert not (clusters / "000000.remove").exists()
    if empty_remove_file:
        (clusters / "000000.remove").touch()

    for rank in range(2):
        docs = [Document(text=f"document {rank}-{index}", id=f"{rank}-{index}") for index in range(5)]
        block = MinhashDedupFilter(
            str(clusters),
            load_cluster_ids=load_ids,
            load_cluster_sizes=load_sizes,
            lines_to_buffer=lines_to_buffer,
        )
        kept = list(block.run(docs, rank=rank, world_size=2))
        expected_indices = list(range(5)) if rank == 0 else [1, 3, 4]
        assert [doc.id for doc in kept] == [f"{rank}-{index}" for index in expected_indices]
        for index, doc in zip(expected_indices, kept):
            cluster_id = {1: 0, 3: 1}.get(index, -1) if rank == 0 else -1
            expected = {}
            if load_ids:
                expected["minhash_cluster_id"] = cluster_id
            if load_sizes:
                expected["minhash_cluster_size"] = 2 if cluster_id != -1 else 1
            assert doc.metadata == expected


@pytest.mark.parametrize("load_ids, load_sizes", [(False, False), (True, False), (False, True), (True, True)])
def test_rank_without_any_cluster_files_keeps_existing_behavior(
    tmp_path: Path, load_ids: bool, load_sizes: bool
) -> None:
    """Keep untouched documents when the rank has no deduplication artifacts at all."""
    docs = [Document(text="unique document", id="unique", metadata={"source": "fixture"})]
    kept = list(MinhashDedupFilter(str(tmp_path), load_cluster_ids=load_ids, load_cluster_sizes=load_sizes).run(docs))
    assert kept == docs
    assert kept[0].metadata == {"source": "fixture"}


@pytest.mark.parametrize("missing_suffix", ["clusters", "sizes"])
def test_missing_requested_metadata_is_not_silently_ignored(tmp_path: Path, missing_suffix: str) -> None:
    """Keep reporting incomplete metadata when a rank has a removal file."""
    (tmp_path / "000000.remove").write_bytes(struct.pack("<I", 0))
    present_suffix = "sizes" if missing_suffix == "clusters" else "clusters"
    (tmp_path / f"000000.{present_suffix}").write_bytes(struct.pack("<2I", 0, 1))
    docs = [Document(text="duplicate document", id="duplicate")]
    with pytest.raises(FileNotFoundError):
        list(MinhashDedupFilter(str(tmp_path), load_cluster_ids=True, load_cluster_sizes=True).run(docs))
