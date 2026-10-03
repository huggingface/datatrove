"""Exercise dataset staging with real Parquet files and mocked Hub APIs."""

import gc
import json
import pickle
import stat
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from fsspec.implementations.local import LocalFileSystem

import datatrove.pipeline.writers.huggingface as hf
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
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


@pytest.mark.parametrize("serialization", ["deepcopy", "pickle", "dill"])
@pytest.mark.parametrize("temporary", [False, True])
@require_pyarrow
def test_staging_ownership_after_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serialization: str, temporary: bool
) -> None:
    """Cloned writers own private staging independently of the original writer."""
    import dill

    for name in ["create_repo", "preupload_lfs_files", "create_commit"]:
        monkeypatch.setattr(hf, name, Mock())
    options = {} if temporary else {"local_working_dir": str(tmp_path / "explicit")}
    original = hf.HuggingFaceDatasetWriter("org/test", cleanup=False, max_file_size=-1, **options)
    original_path = Path(original.local_working_dir.path)
    if serialization == "deepcopy":
        restored = deepcopy(original)
    else:
        serializer = pickle if serialization == "pickle" else dill
        restored = serializer.loads(serializer.dumps(original))
    restored_path = Path(restored.local_working_dir.path)
    assert restored.output_folder is restored.local_working_dir
    assert restored.output_mg.fs is restored.output_folder
    if temporary:
        assert restored_path != original_path
        assert stat.S_IMODE(restored_path.stat().st_mode) == 0o700
    else:
        assert restored_path == original_path
    del original
    gc.collect()
    if temporary:
        assert not original_path.exists()
        assert restored_path.is_dir()
    list(restored.run([Document(text="private staging fixture", id="fixture")]))
    assert (restored_path / "data/00000.parquet").is_file()
    if temporary:
        assert stat.S_IMODE(restored_path.stat().st_mode) == 0o700
    del restored
    gc.collect()
    assert restored_path.exists() is (not temporary)


@require_pyarrow
def test_explicit_staging_from_legacy_pickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load explicit-directory configurations saved before temporary ownership was added."""
    for name in ["create_repo", "preupload_lfs_files", "create_commit"]:
        monkeypatch.setattr(hf, name, Mock())

    def legacy_state(writer: hf.HuggingFaceDatasetWriter) -> dict[str, Any]:
        """Represent the previous writer's state, which had no temporary-directory owner."""
        return {key: value for key, value in writer.__dict__.items() if key != "_local_working_tmpdir"}

    original = hf.HuggingFaceDatasetWriter("org/test", local_working_dir=str(tmp_path), max_file_size=-1)
    with monkeypatch.context() as legacy:
        legacy.setattr(hf.HuggingFaceDatasetWriter, "__getstate__", legacy_state)
        payload = pickle.dumps(original)
    restored = pickle.loads(payload)
    assert restored.local_working_dir.path == str(tmp_path)
    assert restored._local_working_tmpdir is None
    assert list(restored.run([Document(text="legacy fixture", id="0")]))[0].text == "legacy fixture"


def _offline_documents(data: Any, rank: int, world_size: int, destination: str) -> list[Document]:
    """Install offline Hub stand-ins inside spawned workers before writing."""
    import pyarrow.parquet as pq

    def preupload(_dataset: str, additions: list[Any], **_kwargs: Any) -> None:
        """Record real uploaded rows without contacting the Hub."""
        for addition in additions:
            rows = pq.read_table(addition.path_or_fileobj).to_pylist()
            output = Path(destination) / f"{rank}.json"
            output.write_text(json.dumps(rows), encoding="utf-8")

    hf.create_repo = Mock()
    hf.preupload_lfs_files = preupload
    hf.create_commit = Mock()
    return [Document(text=f"rank {rank} document {index}", id=f"{rank}-{index}") for index in range(2)]


@pytest.mark.parametrize("workers", [1, 2])
@require_pyarrow
def test_default_staging_in_local_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int) -> None:
    """Run real sequential and spawned ranks without making any Hub requests."""
    for name in ["create_repo", "preupload_lfs_files", "create_commit"]:
        monkeypatch.setattr(hf, name, Mock())
    executor = LocalPipelineExecutor(
        pipeline=[
            partial(_offline_documents, destination=str(tmp_path)),
            hf.HuggingFaceDatasetWriter("org/test", max_file_size=-1),
        ],
        tasks=2,
        workers=workers,
        start_method="spawn",
        logging_dir=str(tmp_path / "logs"),
    )
    executor.run()
    for rank in range(2):
        assert executor.is_rank_completed(rank)
        rows = json.loads((tmp_path / f"{rank}.json").read_text(encoding="utf-8"))
        assert [(row["id"], row["text"]) for row in rows] == [
            (f"{rank}-{index}", f"rank {rank} document {index}") for index in range(2)
        ]


@pytest.mark.parametrize("executor_kind", ["local", "slurm", "jobs"])
@require_pyarrow
def test_executor_staging_serialization(tmp_path: Path, executor_kind: str) -> None:
    """Exercise the coordinator's deepcopy/dill sequence without submitting remote jobs."""
    import dill

    from datatrove.executor.jobs import JobsPipelineExecutor
    from datatrove.executor.slurm import SlurmPipelineExecutor

    writer = hf.HuggingFaceDatasetWriter("org/test")
    options = {"pipeline": [writer], "tasks": 2, "logging_dir": str(tmp_path / "logs")}
    if executor_kind == "local":
        executor = LocalPipelineExecutor(**options)
    elif executor_kind == "slurm":
        executor = SlurmPipelineExecutor(**options, time="00:01:00", partition="offline-test")
    else:
        options["logging_dir"] = "memory://offline-jobs-logs"
        executor = JobsPipelineExecutor(**options)
    copied = deepcopy(executor)
    restored = dill.loads(dill.dumps(copied, fmode=dill.CONTENTS_FMODE))
    paths = [Path(instance.pipeline[0].local_working_dir.path) for instance in [executor, copied, restored]]
    assert len(set(paths)) == 3
    for path in paths:
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    del writer, executor, copied, options
    gc.collect()
    assert not paths[0].exists()
    assert not paths[1].exists()
    assert paths[2].is_dir()
    del restored
    gc.collect()
    assert not paths[2].exists()
