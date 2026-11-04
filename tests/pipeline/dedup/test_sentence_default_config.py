from pathlib import Path

import pytest

from datatrove.data import Document
from datatrove.pipeline.dedup.sentence_dedup import SentDedupConfig, SentenceDedupSignature


pytest.importorskip("nltk")
pytest.importorskip("xxhash")


@pytest.mark.parametrize("explicit_none", [False, True])
@pytest.mark.parametrize("finder_workers", [1, 3])
def test_default_config_writes_the_same_signatures_as_explicit_config(
    tmp_path: Path, explicit_none: bool, finder_workers: int
) -> None:
    """The optional configuration must use the historical default signature format."""
    implicit_folder = tmp_path / "implicit"
    explicit_folder = tmp_path / "explicit"
    options = {"config": None} if explicit_none else {}
    implicit = SentenceDedupSignature(str(implicit_folder), finder_workers=finder_workers, **options)
    explicit = SentenceDedupSignature(str(explicit_folder), finder_workers=finder_workers, config=SentDedupConfig())
    docs = [
        Document(text="First sentence. Second sentence! Third sentence? Fourth sentence.", id="first"),
        Document(text="First sentence. Second sentence! Third sentence? Different ending.", id="second"),
        Document(text="Too short.", id="short"),
    ]
    assert implicit.config == SentDedupConfig()
    assert implicit.get_hashes(docs[0], 0) == explicit.get_hashes(docs[0], 0)
    assert len(implicit.get_hashes(docs[0], 0)) == 2
    assert implicit.get_hashes(docs[-1], 2) == []
    for rank in range(2):
        implicit.run(docs, rank=rank, world_size=2)
        explicit.run(docs, rank=rank, world_size=2)
    implicit_files = {
        path.relative_to(implicit_folder): path.read_bytes() for path in implicit_folder.rglob("*") if path.is_file()
    }
    explicit_files = {
        path.relative_to(explicit_folder): path.read_bytes() for path in explicit_folder.rglob("*") if path.is_file()
    }
    assert implicit_files
    assert any(implicit_files.values())
    assert implicit_files == explicit_files
