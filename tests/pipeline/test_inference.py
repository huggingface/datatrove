import asyncio
import json
import socket
import tempfile
import threading
from contextlib import contextmanager
from functools import partial
from http.server import HTTPServer
from pathlib import Path

import pytest

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.inference.servers.dummy_server import DummyHandler, DummyServer
from datatrove.pipeline.inference.types import ServerError
from datatrove.pipeline.writers import JsonlWriter


class ControlledRollout:
    """Rollout function that can be configured to fail at specific document IDs or after a certain count."""

    def __init__(self, fail_at_ids=None, fail_after_count=None):
        self.fail_at_ids = fail_at_ids or set()
        self.fail_after_count = fail_after_count
        self.processed_count = 0

    async def __call__(self, document, generate):
        self.processed_count += 1

        if self.fail_after_count and self.processed_count > self.fail_after_count:
            raise RuntimeError(f"Simulated failure after processing {self.fail_after_count} documents")

        if document.id in self.fail_at_ids:
            raise RuntimeError(f"Simulated failure for document {document.id}")

        result = await generate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": document.text},
                        ],
                    }
                ],
                "max_tokens": 100,
            }
        )

        return {
            "text": result.text,
            "finish_reason": result.finish_reason,
            "usage": result.usage,
        }


def test_inference_config_sets_default_concurrency():
    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=4096,
        metric_interval=60,
        rollouts_per_document=3,
        max_concurrent_generations=8,
        max_concurrent_documents=None,
    )

    assert config.max_concurrent_documents == 2


def test_multiple_rollouts_collect_results(tmp_path):
    output_dir = tmp_path / "multi_rollouts"
    documents = [Document(text="hello world", id="multi-1")]

    async def multi_rollout(document, generate, **kwargs):
        await asyncio.sleep(0)
        return "multi-result"

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=3,
        max_concurrent_generations=3,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=multi_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert doc.metadata["rollout_results"] == ["multi-result", "multi-result", "multi-result"]

    output_file = output_dir / "00000.jsonl"
    assert output_file.exists()
    saved = json.loads(output_file.read_text().strip())
    assert saved["metadata"]["rollout_results"] == ["multi-result", "multi-result", "multi-result"]


def test_custom_metadata_key(tmp_path):
    output_dir = tmp_path / "custom_metadata"
    documents = [Document(text="hello", id="custom-1")]

    async def custom_rollout(document, generate, **kwargs):
        return {"value": document.id}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=custom_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
        metadata_key="custom_results",
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert "rollout_results" not in doc.metadata
    assert doc.metadata["custom_results"] == [{"value": "custom-1"}]

    output_file = output_dir / "00000.jsonl"
    assert output_file.exists()
    saved = json.loads(output_file.read_text().strip())
    assert "rollout_results" not in saved["metadata"]
    assert saved["metadata"]["custom_results"] == [{"value": "custom-1"}]


def test_chunked_checkpoint_requires_chunk_index(tmp_path):
    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    with pytest.raises(ValueError, match="chunk_index"):
        InferenceRunner(
            rollout_fn=lambda document, generate, **kwargs: generate({}),
            config=config,
            output_writer=JsonlWriter(
                str(tmp_path / "no_chunk"),
                output_filename="${rank}.jsonl",
                compression=None,
            ),
            checkpoints_local_dir=str(tmp_path / "checkpoints"),
            records_per_chunk=10,
        )

    try:
        InferenceRunner(
            rollout_fn=lambda document, generate, **kwargs: generate({}),
            config=config,
            output_writer=JsonlWriter(
                str(tmp_path / "with_chunk"),
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
                compression=None,
            ),
            checkpoints_local_dir=str(tmp_path / "checkpoints_ok"),
            records_per_chunk=10,
        )
    except ValueError as exc:  # pragma: no cover - explicit failure message
        pytest.fail(f"InferenceRunner should allow chunk_index templates: {exc}")


def test_rollout_handles_multiple_parts(tmp_path):
    parts = ["first chunk", "second chunk", "third chunk"]

    async def chunked_rollout(document, generate, **kwargs):
        document.metadata["rollout_calls"] = document.metadata.get("rollout_calls", 0) + 1
        document.metadata.setdefault("parts_served", [])

        generations = []
        previous_generation = ""

        for index, part in enumerate(parts):
            document.metadata["parts_served"].append(index)
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Process part {index}: {part}\nPrevious: {previous_generation}",
                    }
                ],
                "max_tokens": 128,
            }
            result = await generate(payload)
            previous_generation = result.text
            generations.append(result.text)

        return {"parts": generations}

    output_dir = tmp_path / "callback_output"
    documents = [Document(text="dummy", id="callback-doc")]

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=4096,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=chunked_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert doc.metadata["rollout_calls"] == 1
    assert doc.metadata["parts_served"] == [0, 1, 2]
    assert len(doc.metadata["rollout_results"]) == 1
    assert len(doc.metadata["rollout_results"][0]["parts"]) == len(parts)

    output_file = output_dir / "00000.jsonl"
    assert output_file.exists(), "Expected output document to be saved"
    with output_file.open() as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 1, "Document should be written once after callbacks finish"
    saved_doc = json.loads(lines[0])
    assert saved_doc["id"] == "callback-doc"
    assert len(saved_doc["metadata"]["rollout_results"][0]["parts"]) == len(parts)


def test_query_builder_none_payload_skips_document(tmp_path):
    output_dir = tmp_path / "none_payload_output"
    documents = [Document(text="skip me", id="skip-none")]

    async def none_rollout(document, generate, **kwargs):
        return None

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=none_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert doc.metadata.get("rollout_results") == [], (
        "Document should have no rollout results when rollout returns None"
    )
    output_file = output_dir / "00000.jsonl"
    assert not output_file.exists() or output_file.read_text().strip() == "", (
        "No output should be written when rollout returns None"
    )


def test_async_query_builder_none_payload_skips_document(tmp_path):
    output_dir = tmp_path / "none_async_output"
    documents = [Document(text="skip me async", id="skip-async")]

    async def none_async_rollout(document, generate, **kwargs):
        await asyncio.sleep(0)
        return None

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=none_async_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert doc.metadata.get("rollout_results") == [], (
        "Document should have no rollout results when async rollout returns None"
    )
    output_file = output_dir / "00000.jsonl"
    assert not output_file.exists() or output_file.read_text().strip() == "", (
        "No output should be written when rollout returns None"
    )


def read_output_files(output_path):
    """Helper to read all output files and return document data"""
    output_path = Path(output_path)
    # Look for files matching the pattern used by JsonlWriter
    output_files = sorted(output_path.glob("*_chunk_*.jsonl"))
    all_docs = []

    for output_file in output_files:
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    doc_data = json.loads(line.strip())
                    all_docs.append(doc_data)

    return all_docs, output_files


def test_checkpoint_recovery_and_completeness():
    """
    Comprehensive test that verifies:
    1. Checkpoint creation when pipeline fails mid-execution
    2. Successful recovery and resumption from checkpoints
    3. Complete and correct output after recovery
    4. Proper chunking behavior
    """
    num_docs = 35
    records_per_chunk = 10  # Should create 4 chunks: [0-9], [10-19], [20-29], [30-34]
    fail_after_docs = 22  # Fail after processing 22 docs (middle of chunk 2)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output"
        checkpoint_path = Path(temp_dir) / "checkpoints"

        # Create test documents
        def make_documents():
            return [Document(text=f"Test document {i}", id=str(i)) for i in range(num_docs)]

        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="test-model",
            model_max_context=8192,
            metric_interval=120,
            rollouts_per_document=1,
            max_concurrent_generations=2,  # Low concurrency to ensure predictable chunk completion
            max_concurrent_documents=2,
        )

        # === FIRST RUN: Should fail partway through ===
        failing_rollout = ControlledRollout(fail_after_count=fail_after_docs)

        def make_runner(rollout_fn):
            return InferenceRunner(
                rollout_fn=rollout_fn,
                config=config,
                records_per_chunk=records_per_chunk,
                checkpoints_local_dir=str(checkpoint_path),
                output_writer=JsonlWriter(
                    str(output_path),
                    output_filename="${rank}_chunk_${chunk_index}.jsonl",
                    compression=None,
                ),
            )

        failing_runner = make_runner(failing_rollout)

        # Run first pass - should fail due to rollout exception
        try:
            failing_runner.run(make_documents(), rank=0, world_size=1)
            assert False, "Expected pipeline to fail, but it completed successfully"
        except Exception as e:
            # Pipeline should fail when query_builder raises exceptions
            print(f"Pipeline failed as expected: {e}")

        # === VERIFY CHECKPOINT STATE ===
        assert checkpoint_path.exists(), "Checkpoint directory should exist after failure"

        # Check checkpoint files exist
        checkpoint_files = list(checkpoint_path.rglob("chunk_*.jsonl"))
        assert len(checkpoint_files) > 0, "Should have checkpoint files after partial processing"

        # Check last_chunk tracking file
        last_chunk_file = checkpoint_path / "last_chunk" / "00000.txt"
        if last_chunk_file.exists():
            with open(last_chunk_file, "r") as f:
                last_completed_chunk = int(f.read().strip())
                assert last_completed_chunk >= 0, "Should have completed at least one chunk"

        # Verify partial output exists
        partial_docs, partial_files = read_output_files(output_path)
        assert len(partial_docs) > 0, "Should have some processed documents from first run"
        assert len(partial_docs) <= fail_after_docs, f"Should not have more than {fail_after_docs} docs from first run"

        # === SECOND RUN: Should resume from checkpoint ===
        success_rollout = ControlledRollout()  # No failures this time

        success_runner = make_runner(success_rollout)

        # Run second pass - should complete successfully
        success_runner.run(make_documents(), rank=0, world_size=1)

        # === VERIFY COMPLETE OUTPUT ===
        final_docs, final_files = read_output_files(output_path)

        # Check total document count
        assert len(final_docs) == num_docs, f"Expected {num_docs} documents, got {len(final_docs)}"

        # Check all document IDs are present and unique
        final_ids = {doc["id"] for doc in final_docs}
        expected_ids = {str(i) for i in range(num_docs)}
        assert final_ids == expected_ids, (
            f"Missing IDs: {expected_ids - final_ids}, Extra IDs: {final_ids - expected_ids}"
        )

        # Verify no duplicates (each document processed exactly once)
        final_ids_list = [doc["id"] for doc in final_docs]
        assert len(final_ids_list) == len(set(final_ids_list)), "Found duplicate documents in output"

        # === VERIFY CHUNKING ===
        expected_chunks = (num_docs + records_per_chunk - 1) // records_per_chunk
        assert len(final_files) == expected_chunks, f"Expected {expected_chunks} chunk files, got {len(final_files)}"

        # Verify chunk sizes
        for i, output_file in enumerate(final_files):
            with open(output_file, "r") as f:
                chunk_docs = [json.loads(line.strip()) for line in f if line.strip()]

            if i < expected_chunks - 1:  # All chunks except last should be full
                assert len(chunk_docs) == records_per_chunk, (
                    f"Chunk {i} should have {records_per_chunk} docs, got {len(chunk_docs)}"
                )
            else:  # Last chunk may be partial
                expected_last_chunk_size = num_docs - (expected_chunks - 1) * records_per_chunk
                assert len(chunk_docs) == expected_last_chunk_size, (
                    f"Last chunk should have {expected_last_chunk_size} docs, got {len(chunk_docs)}"
                )

        # === VERIFY INFERENCE RESULTS ===
        for doc in final_docs:
            assert "metadata" in doc, f"Document {doc['id']} missing metadata"
            assert "rollout_results" in doc["metadata"], f"Document {doc['id']} missing rollout_results"

            rollout_results = doc["metadata"]["rollout_results"]
            assert len(rollout_results) > 0, f"Document {doc['id']} has no rollout results"

            # Verify rollout result structure (dummy server should return success)
            for result in rollout_results:
                assert "text" in result, f"Rollout result missing 'text' field for doc {doc['id']}"
                assert "finish_reason" in result, f"Rollout result missing 'finish_reason' field for doc {doc['id']}"
                assert "usage" in result, f"Rollout result missing 'usage' field for doc {doc['id']}"


def test_complete_pipeline_with_various_scenarios():
    """
    Test complete pipeline execution matching the original bug scenario:
    1. 1005 documents (matches original bug report)
    2. 500 documents per chunk (matches original bug report)
    3. High concurrency to stress-test the fix
    4. Validates all documents are saved (vs original bug where only ~7 were saved)
    """
    num_docs = 1005
    records_per_chunk = 500  # Creates 3 chunks: [0-499], [500-999], [1000-1004]

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output"
        checkpoint_path = Path(temp_dir) / "checkpoints"
        logs_path = Path(temp_dir) / "logs"

        # Create test documents matching original bug scenario
        documents = [Document(text="What's the weather in Tokyo?", id=str(i)) for i in range(num_docs)]

        # Normal query builder that doesn't cause pipeline failures
        async def normal_rollout(document, generate, **kwargs):
            result = await generate(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": document.text},
                            ],
                        }
                    ],
                    "max_tokens": 4096,
                }
            )
            return {
                "text": result.text,
                "finish_reason": result.finish_reason,
                "usage": result.usage,
            }

        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="reducto/RolmOCR",
            model_max_context=8192,
            metric_interval=120,
            rollouts_per_document=1,
            max_concurrent_generations=10,  # Reduced from 500 for CI compatibility
            max_concurrent_documents=10,  # Reduced from 500 for CI compatibility
        )

        pipeline_executor = LocalPipelineExecutor(
            pipeline=[
                documents,
                InferenceRunner(
                    rollout_fn=normal_rollout,
                    config=config,
                    records_per_chunk=records_per_chunk,
                    checkpoints_local_dir=str(checkpoint_path),
                    output_writer=JsonlWriter(
                        str(output_path),
                        output_filename="${rank}_chunk_${chunk_index}.jsonl",
                        compression=None,
                    ),
                ),
            ],
            logging_dir=str(logs_path / "complete_test_run"),
            tasks=1,
        )

        # Run pipeline - should complete successfully and save ALL documents
        pipeline_executor.run()

        # === VERIFY COMPLETE PROCESSING ===
        final_docs, final_files = read_output_files(output_path)

        # This is the key test - ALL documents should be processed (original bug only saved ~7)
        assert len(final_docs) == num_docs, f"Expected {num_docs} documents, got {len(final_docs)}"

        # Verify all document IDs present
        processed_ids = {doc["id"] for doc in final_docs}
        expected_ids = {str(i) for i in range(num_docs)}
        assert processed_ids == expected_ids, "Not all documents were processed"

        # === VERIFY SUCCESSFUL RESULTS ===
        for doc in final_docs:
            rollout_results = doc["metadata"]["rollout_results"]
            assert len(rollout_results) > 0, f"Document {doc['id']} has no rollout results"

            # All results should be successful (dummy server always succeeds)
            for result in rollout_results:
                assert "text" in result, "Success result should have text"
                assert "finish_reason" in result, "Success result should have finish_reason"
                assert "usage" in result, "Success result should have usage stats"
                assert "error" not in result, "Should not have error in successful result"

        # === VERIFY CHUNKING CORRECTNESS ===
        expected_chunks = (num_docs + records_per_chunk - 1) // records_per_chunk  # Should be 3 chunks
        assert len(final_files) == expected_chunks, f"Expected {expected_chunks} chunks, got {len(final_files)}"

        # Verify chunk contents
        chunk_doc_counts = []
        for output_file in final_files:
            with open(output_file, "r") as f:
                chunk_docs = [json.loads(line.strip()) for line in f if line.strip()]
                chunk_doc_counts.append(len(chunk_docs))

        # First two chunks should have exactly 500 documents each
        for i, count in enumerate(chunk_doc_counts[:-1]):
            assert count == records_per_chunk, f"Chunk {i} should have {records_per_chunk} docs, got {count}"

        # Last chunk should have remaining 5 documents (1000-1004)
        expected_last_count = num_docs - (expected_chunks - 1) * records_per_chunk  # Should be 5
        assert chunk_doc_counts[-1] == expected_last_count, (
            f"Last chunk should have {expected_last_count} docs, got {chunk_doc_counts[-1]}"
        )


def test_shared_context_as_dict(tmp_path):
    """Test that shared_context as a dict passes kwargs to rollout_fn."""
    output_dir = tmp_path / "shared_context_dict"
    documents = [Document(text="test", id="shared-1")]

    async def rollout_with_context(document, generate, custom_value=None, another_param=None):
        assert custom_value == "test_value", "custom_value should be passed from shared_context"
        assert another_param == 42, "another_param should be passed from shared_context"
        result = await generate(
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
                "max_tokens": 100,
            }
        )
        return {"custom_value": custom_value, "another_param": another_param, "result": result.text}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=rollout_with_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
        shared_context={"custom_value": "test_value", "another_param": 42},
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert doc.metadata["rollout_results"][0]["custom_value"] == "test_value"
    assert doc.metadata["rollout_results"][0]["another_param"] == 42


def test_shared_context_as_callable(tmp_path):
    """Test that shared_context as a callable passes kwargs to rollout_fn."""
    output_dir = tmp_path / "shared_context_callable"
    documents = [Document(text="test", id="shared-2")]

    call_count = {"count": 0}

    def make_shared_context():
        call_count["count"] += 1
        return {"dynamic_value": f"value_{call_count['count']}"}

    async def rollout_with_context(document, generate, dynamic_value=None):
        assert dynamic_value is not None, "dynamic_value should be passed from shared_context"
        await generate(
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
                "max_tokens": 100,
            }
        )
        return {"dynamic_value": dynamic_value}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=rollout_with_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
        shared_context=make_shared_context,
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert doc.metadata["rollout_results"][0]["dynamic_value"] == "value_1"
    assert call_count["count"] == 1, "shared_context callable should be called once"


def test_shared_context_as_context_manager(tmp_path):
    """Test that shared_context as a context manager properly manages resources."""
    output_dir = tmp_path / "shared_context_cm"
    documents = [Document(text="test", id="shared-3")]

    cleanup_called = {"called": False}

    class TestContextManager:
        def __init__(self):
            self.value = "context_value"
            self.entered = False

        def __enter__(self):
            self.entered = True
            return {"context_value": self.value}

        def __exit__(self, exc_type, exc_val, exc_tb):
            cleanup_called["called"] = True
            return False

    async def rollout_with_context(document, generate, context_value=None):
        assert context_value == "context_value", "context_value should be passed from shared_context"
        await generate(
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
                "max_tokens": 100,
            }
        )
        return {"context_value": context_value}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    cm = TestContextManager()
    runner = InferenceRunner(
        rollout_fn=rollout_with_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
        shared_context=cm,
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert doc.metadata["rollout_results"][0]["context_value"] == "context_value"
    assert cm.entered, "Context manager should have been entered"
    assert cleanup_called["called"], "Context manager cleanup should have been called"


def test_shared_context_none(tmp_path):
    """Test that rollout_fn works without shared_context (no kwargs passed)."""
    output_dir = tmp_path / "shared_context_none"
    documents = [Document(text="test", id="shared-4")]

    async def rollout_no_context(document, generate, **kwargs):
        # This should work fine without any kwargs
        result = await generate(
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
                "max_tokens": 100,
            }
        )
        return {"result": result.text}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=rollout_no_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
        shared_context=None,  # Explicitly None
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert "result" in doc.metadata["rollout_results"][0]


def test_shared_context_callable_returns_context_manager(tmp_path):
    """Test that a callable that returns a context manager works correctly."""
    output_dir = tmp_path / "shared_context_callable_cm"
    documents = [Document(text="test", id="shared-5")]

    cleanup_called = {"called": False}

    @contextmanager
    def test_context_manager(value: str):
        cleanup_called["called"] = False
        try:
            yield {"test_value": value}
        finally:
            cleanup_called["called"] = True

    async def rollout_with_context(document, generate, test_value=None):
        assert test_value == "test_value", "test_value should be passed from shared_context"
        await generate(
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
                "max_tokens": 100,
            }
        )
        return {"test_value": test_value}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    # Test 1: Callable that returns a context manager (using partial)
    cleanup_called["called"] = False
    runner1 = InferenceRunner(
        rollout_fn=rollout_with_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir / "test1"), output_filename="${rank}.jsonl", compression=None),
        shared_context=partial(test_context_manager, "test_value"),
    )

    asyncio.run(runner1.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert doc.metadata["rollout_results"][0]["test_value"] == "test_value"
    assert cleanup_called["called"], "Context manager cleanup should have been called (callable version)"

    # Test 2: Direct context manager (calling the function and passing the result)
    cleanup_called["called"] = False
    runner2 = InferenceRunner(
        rollout_fn=rollout_with_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir / "test2"), output_filename="${rank}.jsonl", compression=None),
        shared_context=test_context_manager("test_value"),
    )

    asyncio.run(runner2.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert doc.metadata["rollout_results"][0]["test_value"] == "test_value"
    assert cleanup_called["called"], "Context manager cleanup should have been called (direct version)"


def test_endpoint_server(tmp_path):
    """Test EndpointServer with a mock HTTP server."""
    output_dir = tmp_path / "endpoint_test"
    documents = [Document(text="hello endpoint", id="endpoint-1")]

    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    # Start a simple HTTP server with DummyHandler
    server = HTTPServer(("localhost", port), DummyHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    # Give the server a moment to start
    asyncio.run(asyncio.sleep(0.1))

    try:

        async def endpoint_rollout(document, generate):
            result = await generate(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": document.text}],
                        }
                    ],
                    "max_tokens": 100,
                }
            )
            return {
                "text": result.text,
                "finish_reason": result.finish_reason,
                "usage": result.usage,
            }

        config = InferenceConfig(
            server_type="endpoint",
            model_name_or_path="test-model",
            model_max_context=2048,
            endpoint_url=f"http://localhost:{port}",
            metric_interval=60,
            rollouts_per_document=1,
            max_concurrent_generations=1,
            max_concurrent_documents=None,
        )

        runner = InferenceRunner(
            rollout_fn=endpoint_rollout,
            config=config,
            output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
        )

        asyncio.run(runner.run_async(documents, rank=0))

        doc = documents[0]
        assert "rollout_results" in doc.metadata
        assert len(doc.metadata["rollout_results"]) == 1
        assert "text" in doc.metadata["rollout_results"][0]

        output_file = output_dir / "00000.jsonl"
        assert output_file.exists()
        saved = json.loads(output_file.read_text().strip())
        assert saved["metadata"]["rollout_results"][0]["text"] == doc.metadata["rollout_results"][0]["text"]
    finally:
        server.shutdown()
        server.server_close()


def test_simple_startup_and_cleanup():
    """
    Scenario 1: Simple scenario - check that resources are correctly deallocated.

    Verifies:
    - Server starts successfully
    - Server becomes ready
    - Server can handle requests
    - All resources are properly cleaned up on exit
    """

    async def run_test():
        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="dummy",
        )
        server = DummyServer(config, rank=0)

        async with server:
            await server._server_ready

            print("Server is ready! Verifying port...")
            # Verify port was assigned
            assert server._port is not None
            assert server._port >= 3000
            assert server._port <= 65535

            print("Sending test request...")
            # Send a test request
            response = await server.make_request({"messages": [{"role": "user", "content": "test"}]})
            assert "dummy text content" in response["choices"][0]["message"]["content"]
            print("Test assertions passed, exiting context manager...")

        # Verify server process is cleaned up
        if server._server_process:
            assert server._server_process.returncode is not None

        # Verify monitoring task is cancelled
        if server._server_monitoring_task:
            assert server._server_monitoring_task.cancelled() or server._server_monitoring_task.done()

        # Verify background start task is cancelled
        assert server._bg_start_server_task.cancelled() or server._bg_start_server_task.done()

    asyncio.run(run_test())


def test_server_auto_restart_turn_off():
    async def run_test():
        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="dummy",
        )
        server = DummyServer(config, rank=0)

        async with server:
            await server._server_ready

            # kill the server process
            server.kill_server()

            # Ensure we have enough time for server to notice
            await asyncio.sleep(3)

            # Verify requests work
            with pytest.raises(ServerError):
                await server.make_request({"messages": [{"role": "user", "content": "test"}]})

    asyncio.run(run_test())


class FailingDummyServer(DummyServer):
    def __init__(self, config, rank):
        super().__init__(config, rank)
        self.start_attempts = 0

    async def start_server(self):
        self.start_attempts += 1
        raise Exception("Simulated start failure")

    async def _wait_until_ready(self, max_attempts=1, delay_sec=0.1):
        # Override to avoid waiting in tests
        pass


def test_server_retries():
    config = InferenceConfig(
        server_type="dummy", model_name_or_path="test-model", model_max_context=100, metric_interval=60
    )

    server = None

    # We expect an exception because it fails every time
    async def run_test():
        nonlocal server
        server = FailingDummyServer(config, rank=0)
        # "async open the server"
        async with server:
            # Wait until it fails
            with pytest.raises(ServerError):
                await server.make_request({})

    asyncio.run(run_test())

    # Should attempt 2 times: initial attempt (0) + 1 retry (1) => loop runs for 0, 1.
    assert server.start_attempts == 2


def test_custom_server(tmp_path):
    """Test CustomServer with a small python script."""
    output_dir = tmp_path / "custom_test"
    documents = [Document(text="hello custom", id="custom-1")]

    # Create the server script
    script_path = tmp_path / "server_script.py"
    script_path.write_text(
        """
import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys

class SimpleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            response = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from custom server!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10}
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
             self.send_error(404)

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"object": "list", "data": []}).encode('utf-8'))
        else:
            self.send_error(404)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args, unknown = parser.parse_known_args()
    print("Application startup complete", flush=True)
    HTTPServer(('127.0.0.1', args.port), SimpleHandler).serve_forever()
"""
    )

    async def custom_rollout(document, generate):
        result = await generate({"messages": [{"role": "user", "content": document.text}]})
        return {"text": result.text}

    config = InferenceConfig(
        server_type="custom",
        model_name_or_path="test-model",
        model_max_context=2048,
        model_kwargs={"server_script": str(script_path)},
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
    )

    runner = InferenceRunner(
        rollout_fn=custom_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    rollout_results = doc.metadata["rollout_results"]
    first_result = rollout_results[0]
    assert isinstance(first_result, dict)
    assert first_result["text"] == "Hello from custom server!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
