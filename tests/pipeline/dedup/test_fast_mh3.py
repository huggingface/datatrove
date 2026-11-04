import json
import os
import struct
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from datatrove.data import Document
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupCluster, MinhashDedupFilter
from datatrove.pipeline.writers.jsonl import JsonlWriter


EDGES = [(2, 2, 1, 0), (1, 0, 0, 0), (1, 1, 0, 2), (0, 2, 2, 0), (2, 2, 0, 0)]
SENTINEL = 2**32 - 1


@pytest.fixture
def binaries() -> Path:
    """Locate explicitly built binaries, keeping Cargo out of ordinary Python tests."""
    directory = os.environ.get("DATATROVE_FAST_MH3_BIN_DIR")
    if directory is None:
        pytest.skip("Build fast_mh3 and set DATATROVE_FAST_MH3_BIN_DIR to run Rust integration tests")
    path = Path(directory).resolve()
    for name in ("local", "s3"):
        assert (path / name).is_file(), f"Missing Rust binary: {path / name}"
    return path


def _run(
    binary: Path,
    inputs: str,
    output: str,
    total_files: int = 1,
    save_ids: bool = True,
    concurrency: int = 1,
    env: dict[str, str] | None = None,
) -> None:
    args = [
        str(binary),
        "--input-folder",
        inputs,
        "--output-folder",
        output,
        "--total-files",
        str(total_files),
        "--concurrent-ops" if binary.name == "local" else "--downloads",
        str(concurrency),
    ]
    if save_ids:
        args.append("--save-cluster-id")
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def _reference(path: Path, edges: list[tuple[int, int, int, int]], files: int = 1) -> tuple[Path, dict[str, bytes]]:
    inputs, output = path / "input", path / "python"
    inputs.mkdir()
    for index in range(files):
        (inputs / f"{index:05d}.dups").write_bytes(b"".join(struct.pack("<4I", *edge) for edge in edges[index::files]))
    MinhashDedupCluster(
        str(inputs), str(output), config=MinhashConfig(num_buckets=1), save_cluster_id=True, save_cluster_size=True
    )()
    return inputs, {file.name: file.read_bytes() for file in output.glob("*")}


@pytest.mark.parametrize(
    "edges",
    [
        [],
        EDGES,
        list(reversed(EDGES)),
        [(c, d, a, b) for a, b, c, d in EDGES],
        [(0, 0, 1, 0)],
        [(0, 65536, 1, 16777216)],
        [(SENTINEL, SENTINEL, 0, 1), (0, 1, 2, 3), (1, 2, SENTINEL, SENTINEL), (1, 0, 0, 2)],
    ],
    ids=["empty", "transitive", "reordered", "reversed-ends", "root-only-rank", "large-offsets", "index"],
)
@pytest.mark.parametrize("save_ids", [False, True])
def test_local_outputs(binaries: Path, tmp_path: Path, edges: list[tuple[int, int, int, int]], save_ids: bool) -> None:
    """Compare complete binary outputs with Python, including sentinel and empty cases."""
    inputs, expected = _reference(tmp_path, edges)
    output = tmp_path / "rust"
    _run(binaries / "local", str(inputs), str(output), save_ids=save_ids)
    actual = {file.name: file.read_bytes() for file in output.iterdir()}
    # Rust already creates empty .remove files for ranks containing only roots.
    actual = {name: data for name, data in actual.items() if data or not name.endswith(".remove")}
    if not save_ids:
        expected = {name: data for name, data in expected.items() if not name.endswith(".clusters")}
    assert actual == expected


@pytest.mark.parametrize("concurrency", [0, 1, 3])
def test_parallel_input_order(binaries: Path, tmp_path: Path, concurrency: int) -> None:
    """IDs stay global and match Python when several input files race to update roots."""
    inputs, expected = _reference(tmp_path, EDGES * 3, files=3)
    output = tmp_path / "rust"
    _run(binaries / "local", str(inputs), str(output), total_files=3, concurrency=concurrency)
    for suffix in (".clusters", ".sizes"):
        assert {file.name: file.read_bytes() for file in output.glob(f"*{suffix}")} == {
            name: data for name, data in expected.items() if name.endswith(suffix)
        }


def test_python_filter_metadata(binaries: Path, tmp_path: Path) -> None:
    """The Python filter loads IDs for both survivors and excluded documents."""
    inputs, _ = _reference(tmp_path, EDGES)
    output = tmp_path / "rust"
    _run(binaries / "local", str(inputs), str(output))
    membership = {(0, 0): 0, (1, 0): 0, (2, 2): 0, (0, 2): 1, (1, 1): 1, (2, 0): 1}
    excluded = tmp_path / "excluded"
    expected_excluded = {}
    for rank in range(3):
        documents = [Document(text=f"doc {rank}:{index}", id=f"{rank}:{index}") for index in range(4)]
        survivors = list(
            MinhashDedupFilter(
                str(output),
                load_cluster_ids=True,
                load_cluster_sizes=True,
                exclusion_writer=JsonlWriter(str(excluded), compression=None),
            )(documents, rank=rank)
        )
        for index, doc in enumerate(documents):
            cluster_id = membership.get((rank, index), -1)
            assert doc.metadata["minhash_cluster_id"] == cluster_id
            assert doc.metadata["minhash_cluster_size"] == (3 if cluster_id >= 0 else 1)
        removed = {item[0] for item in struct.iter_unpack("<I", (output / f"{rank:06d}.remove").read_bytes())}
        assert {doc.id for doc in survivors} == {f"{rank}:{index}" for index in range(4) if index not in removed}
        expected_excluded.update({documents[index].id: documents[index].metadata for index in removed})
    records = [json.loads(line) for file in excluded.glob("*.jsonl") for line in file.read_text().splitlines()]
    assert {record["id"]: record["metadata"] for record in records} == expected_excluded


@pytest.fixture
def s3_server() -> Iterator[tuple[Any, dict[str, str]]]:
    """Serve S3 on loopback with dummy credentials, without using the user's AWS configuration."""
    boto3 = pytest.importorskip("boto3")
    server_module = pytest.importorskip("moto.server")
    server = server_module.ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    try:
        host, port = server.get_host_and_port()
        endpoint = f"http://{host}:{port}"
        env = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
        env.update(
            AWS_ACCESS_KEY_ID="testing",
            AWS_SECRET_ACCESS_KEY="testing",
            AWS_REGION="us-east-1",
            AWS_ENDPOINT_URL=endpoint,
            AWS_EC2_METADATA_DISABLED="true",
            AWS_CONFIG_FILE=os.devnull,
            AWS_SHARED_CREDENTIALS_FILE=os.devnull,
        )
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        client.create_bucket(Bucket="test-bucket")
        yield client, env
    finally:
        server.stop()


@pytest.mark.parametrize("save_ids", [False, True])
def test_s3_outputs(binaries: Path, tmp_path: Path, s3_server: tuple[Any, dict[str, str]], save_ids: bool) -> None:
    """Exercise the S3 binary's input, output keys, and finalized cluster records."""
    client, env = s3_server
    inputs, expected = _reference(tmp_path, EDGES)
    client.put_object(Bucket="test-bucket", Key="input/edges.dups", Body=(inputs / "00000.dups").read_bytes())
    _run(binaries / "s3", "s3://test-bucket/input/", "s3://test-bucket/output/", save_ids=save_ids, env=env)
    actual = {}
    for obj in client.list_objects_v2(Bucket="test-bucket", Prefix="output/").get("Contents", []):
        with client.get_object(Bucket="test-bucket", Key=obj["Key"])["Body"] as body:
            actual[obj["Key"].removeprefix("output/")] = body.read()
    if not save_ids:
        expected = {name: data for name, data in expected.items() if not name.endswith(".clusters")}
    assert actual == expected
    assert not client.list_multipart_uploads(Bucket="test-bucket").get("Uploads")


def test_s3_cluster_multipart(binaries: Path, s3_server: tuple[Any, dict[str, str]]) -> None:
    """Cross the 5 MiB boundary and verify the final partial cluster upload part."""
    client, env = s3_server
    count = 5 * 1024 * 1024 // 8 + 1
    edges = b"".join(struct.pack("<4I", 0, 0, 0, doc) for doc in range(1, count))
    client.put_object(Bucket="test-bucket", Key="input/edges.dups", Body=edges)
    _run(binaries / "s3", "s3://test-bucket/input/", "s3://test-bucket/output/", env=env)
    response = client.get_object(Bucket="test-bucket", Key="output/000000.clusters")
    assert response["ETag"].endswith('-2"')
    with response["Body"] as body:
        assert body.read() == b"".join(struct.pack("<II", doc, 0) for doc in range(count))
    assert not client.list_multipart_uploads(Bucket="test-bucket").get("Uploads")
