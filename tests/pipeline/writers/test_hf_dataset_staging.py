"""Exercise dataset staging with real Parquet files and mocked Hub APIs."""

import gc
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from fsspec.implementations.local import LocalFileSystem

import datatrove.pipeline.writers.huggingface as hf
from datatrove.data import Document
from datatrove.io import get_datafolder
from datatrove.pipeline.readers import ParquetReader
from tests.utils import require_pyarrow


@pytest.mark.parametrize("directory_kind", ["omitted", "none", "str", "datafolder", "tuple"])
@pytest.mark.parametrize("cleanup", [False, True])
@require_pyarrow
def test_staging_and_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_kind: str, cleanup: bool
) -> None:
    """Keep staging alive, upload the actual files and preserve two ranks' documents."""
    import pyarrow.parquet as pq

    uploaded: list[tuple[str, str]] = []

    def preupload(_dataset: str, additions: list[Any], **_kwargs: Any) -> None:
        """Read real staged Parquet files before optional local cleanup."""
        for addition in additions:
            rows = pq.read_table(addition.path_or_fileobj).to_pylist()
            uploaded.extend((row["id"], row["text"]) for row in rows)

    def commit(_dataset: str, operations: list[Any], **_kwargs: Any) -> None:
        """Mirror the Hub bookkeeping that rejects reusing a committed addition."""
        for operation in operations:
            assert not getattr(operation, "_is_committed", False)
            operation._is_committed = True

    mocks = {
        "create_repo": Mock(),
        "preupload_lfs_files": Mock(side_effect=preupload),
        "create_commit": Mock(side_effect=commit),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(hf, name, mock)
    folder = tmp_path / "explicit"
    kwargs: dict[str, Any] = {}
    if directory_kind == "none":
        kwargs["local_working_dir"] = None
    elif directory_kind == "str":
        kwargs["local_working_dir"] = str(folder)
    elif directory_kind == "datafolder":
        kwargs["local_working_dir"] = get_datafolder(str(folder))
    elif directory_kind == "tuple":
        kwargs["local_working_dir"] = (str(folder), LocalFileSystem())
    writer = hf.HuggingFaceDatasetWriter(dataset="org/test", cleanup=cleanup, max_file_size=-1, **kwargs)
    staging = Path(writer.local_working_dir.path)
    assert writer.output_folder.path == writer.local_working_dir.path
    if directory_kind in {"omitted", "none"}:
        gc.collect()
        assert staging.is_dir()
    else:
        assert staging == folder

    expected = []
    for rank in [0, 1]:
        data = [Document(text=f"Árbol 🌱 {rank}-{index}", id=f"{rank}-{index}") for index in range(2)]
        with writer:
            for doc in data:
                writer.write(doc, rank=rank)
        expected.extend((doc.id, doc.text) for doc in data)
    assert uploaded == expected
    assert mocks["create_repo"].call_count == 1
    assert mocks["preupload_lfs_files"].call_count == 2
    assert mocks["create_commit"].call_count == 2
    if cleanup:
        assert not list(staging.rglob("*.parquet"))
    else:
        assert [(doc.id, doc.text) for doc in ParquetReader(str(staging)).run()] == expected
