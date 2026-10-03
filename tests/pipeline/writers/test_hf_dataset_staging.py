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
            staging = Path(addition.path_or_fileobj).parents[1]
            (Path(destination) / f"{rank}.staging.json").write_text(
                json.dumps({"path": str(staging), "mode": stat.S_IMODE(staging.stat().st_mode)}), encoding="utf-8"
            )

    hf.create_repo = Mock()
    hf.preupload_lfs_files = preupload
    hf.create_commit = Mock()
    return [Document(text=f"rank {rank} document {index}", id=f"{rank}-{index}") for index in range(2)]


@pytest.mark.parametrize("workers,start_method", [(1, "spawn"), (2, "spawn"), (2, "forkserver")])
@require_pyarrow
def test_default_staging_in_local_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int, start_method: str
) -> None:
    """Run real workers and remove their temporary staging before they exit."""
    for name in ["create_repo", "preupload_lfs_files", "create_commit"]:
        monkeypatch.setattr(hf, name, Mock())
    executor = LocalPipelineExecutor(
        pipeline=[
            partial(_offline_documents, destination=str(tmp_path)),
            hf.HuggingFaceDatasetWriter("org/test", max_file_size=-1),
        ],
        tasks=2,
        workers=workers,
        start_method=start_method,
        logging_dir=str(tmp_path / "logs"),
    )
    executor.run()
    staging_paths = []
    for rank in range(2):
        assert executor.is_rank_completed(rank)
        rows = json.loads((tmp_path / f"{rank}.json").read_text(encoding="utf-8"))
        assert [(row["id"], row["text"]) for row in rows] == [
            (f"{rank}-{index}", f"rank {rank} document {index}") for index in range(2)
        ]
        staging = json.loads((tmp_path / f"{rank}.staging.json").read_text(encoding="utf-8"))
        assert staging["mode"] == 0o700
        staging_paths.append(Path(staging["path"]))
    assert len(set(staging_paths)) == 2
    assert all(not path.exists() for path in staging_paths)


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


@pytest.mark.parametrize("temporary", [False, True])
@pytest.mark.parametrize("cleanup", [False, True])
@require_pyarrow
def test_close_cleanup_and_direct_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, temporary: bool, cleanup: bool
) -> None:
    """Clean owned staging promptly and preserve explicit paths and retention settings."""
    import pyarrow.parquet as pq

    for name in ["create_repo", "create_commit"]:
        monkeypatch.setattr(hf, name, Mock())
    uploaded = []

    def preupload(_dataset: str, additions: list[Any], **_kwargs: Any) -> None:
        """Check private permissions and read real files before each upload finishes."""
        for addition in additions:
            staging = Path(addition.path_or_fileobj).parents[1]
            if temporary:
                assert stat.S_IMODE(staging.stat().st_mode) == 0o700
            uploaded.extend((row["id"], row["text"]) for row in pq.read_table(addition.path_or_fileobj).to_pylist())

    monkeypatch.setattr(hf, "preupload_lfs_files", preupload)
    options = {} if temporary else {"local_working_dir": str(tmp_path)}
    writer = hf.HuggingFaceDatasetWriter("org/test", cleanup=cleanup, max_file_size=-1, **options)
    paths = []
    for rank in [0, 1]:
        writer.write(Document(text=f"rank {rank} Árbol 🌱", id=str(rank)), rank=rank)
        staging = Path(writer.local_working_dir.path)
        paths.append(staging)
        writer.close()
        assert staging.exists() is (not (temporary and cleanup))
        assert len(list(staging.rglob("*.parquet"))) == (0 if cleanup else rank + 1)
        # A second close must not recreate staging or make later reuse lose privacy.
        writer.close()
        assert staging.exists() is (not (temporary and cleanup))
    assert uploaded == [(str(rank), f"rank {rank} Árbol 🌱") for rank in [0, 1]]
    assert (paths[0] != paths[1]) is (temporary and cleanup)
    del writer
    gc.collect()
    if temporary:
        assert all(not path.exists() for path in paths)
    else:
        assert tmp_path.is_dir()


@pytest.mark.parametrize("serialization", ["deepcopy", "pickle", "dill"])
@require_pyarrow
def test_serialize_closed_temporary_writer(monkeypatch: pytest.MonkeyPatch, serialization: str) -> None:
    """Closed configurations restore live private staging before another write."""
    import dill

    for name in ["create_repo", "preupload_lfs_files", "create_commit"]:
        monkeypatch.setattr(hf, name, Mock())
    writer = hf.HuggingFaceDatasetWriter("org/test", max_file_size=-1)
    original_path = Path(writer.local_working_dir.path)
    writer.write(Document(text="first fixture", id="0"))
    writer.close()
    assert not original_path.exists()
    if serialization == "deepcopy":
        restored = deepcopy(writer)
    else:
        serializer = pickle if serialization == "pickle" else dill
        restored = serializer.loads(serializer.dumps(writer))
    path = Path(restored.local_working_dir.path)
    assert path != original_path
    assert path.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    restored.write(Document(text="restored fixture", id="1"))
    assert restored.local_working_dir.path == str(path)
    restored.close()
    assert not path.exists()


@pytest.mark.parametrize("failure_step", ["preupload_lfs_files", "create_commit"])
@require_pyarrow
def test_failed_close_keeps_temporary_staging(monkeypatch: pytest.MonkeyPatch, failure_step: str) -> None:
    """Do not remove the owned directory on upload or commit failure."""
    for name in ["create_repo", "preupload_lfs_files", "create_commit"]:
        monkeypatch.setattr(hf, name, Mock())

    def fail_upload(*_args: Any, **_kwargs: Any) -> None:
        """Raise a fresh error so the mock does not retain a traceback and its writer."""
        raise RuntimeError("offline fixture failure")

    monkeypatch.setattr(hf, failure_step, Mock(side_effect=fail_upload))
    writer = hf.HuggingFaceDatasetWriter("org/test", max_file_size=-1)
    path = Path(writer.local_working_dir.path)
    writer.write(Document(text="failure fixture", id="0"))
    with pytest.raises(RuntimeError, match="offline fixture failure"):
        writer.close()
    assert path.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert (path / "data/00000.parquet").exists() is (failure_step == "preupload_lfs_files")
    del writer
    gc.collect()
    assert not path.exists()


@require_pyarrow
def test_rotation_keeps_staging_until_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rotated uploads leave the directory available until the successful final commit."""
    import pyarrow.parquet as pq

    for name in ["create_repo", "create_commit"]:
        monkeypatch.setattr(hf, name, Mock())
    uploaded = []

    def preupload(_dataset: str, additions: list[Any], **_kwargs: Any) -> None:
        """Read rotated Parquet files before per-file cleanup."""
        for addition in additions:
            uploaded.extend(row["id"] for row in pq.read_table(addition.path_or_fileobj).to_pylist())

    monkeypatch.setattr(hf, "preupload_lfs_files", preupload)
    writer = hf.HuggingFaceDatasetWriter("org/test", max_file_size=1)
    path = Path(writer.local_working_dir.path)
    for index in range(3):
        writer.write(Document(text=f"rotation fixture {index}", id=str(index)))
        assert path.is_dir()
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    writer.close()
    assert uploaded == ["0", "1", "2"]
    assert not path.exists()
