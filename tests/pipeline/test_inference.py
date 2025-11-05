import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.writers import JsonlWriter


class ControlledQueryBuilder:
    """Query builder that can be configured to fail at specific document IDs or after a certain count"""

    def __init__(self, fail_at_ids=None, fail_after_count=None):
        self.fail_at_ids = fail_at_ids or set()
        self.fail_after_count = fail_after_count
        self.processed_count = 0

    def __call__(self, runner, document):
        self.processed_count += 1

        if self.fail_after_count and self.processed_count > self.fail_after_count:
            raise RuntimeError(f"Simulated failure after processing {self.fail_after_count} documents")

        if document.id in self.fail_at_ids:
            raise RuntimeError(f"Simulated failure for document {document.id}")

        return {
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


def test_chunked_checkpoint_requires_chunk_index(tmp_path):
    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        temperature=0.0,
        model_max_context=2048,
        max_concurrent_requests=1,
        max_concurrent_tasks=1,
        metric_interval=60,
    )

    with pytest.raises(ValueError, match="chunk_index"):
        InferenceRunner(
            query_builder=lambda runner, document: {},
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
            query_builder=lambda runner, document: {},
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


def test_callback_recursion_processes_all_parts(tmp_path):
    parts = ["first chunk", "second chunk", "third chunk"]

    def callback_query_builder(runner, document):
        inference_results = document.metadata.get("inference_results") or []
        part_index = min(len(inference_results), len(parts) - 1)
        callback_flag = len(inference_results) < len(parts) - 1

        document.metadata["builder_calls"] = document.metadata.get("builder_calls", 0) + 1
        document.metadata.setdefault("parts_served", []).append(part_index)
        document.metadata.setdefault("callback_flags", []).append(callback_flag)

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Process part {part_index}: {parts[part_index]}",
                        }
                    ],
                }
            ],
            "max_tokens": 128,
            "callback": callback_flag,
        }

    output_dir = tmp_path / "callback_output"
    documents = [Document(text="dummy", id="callback-doc")]

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        temperature=0.0,
        model_max_context=4096,
        max_concurrent_requests=1,
        max_concurrent_tasks=1,
        metric_interval=60,
    )

    runner = InferenceRunner(
        query_builder=callback_query_builder,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0, world_size=1))

    doc = documents[0]
    assert doc.metadata["builder_calls"] == len(parts)
    assert doc.metadata["parts_served"] == [0, 1, 2]
    assert doc.metadata["callback_flags"] == [True, True, False]
    assert len(doc.metadata["inference_results"]) == len(parts)

    output_file = output_dir / "00000.jsonl"
    assert output_file.exists(), "Expected output document to be saved"
    with output_file.open() as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 1, "Document should be written once after callbacks finish"
    saved_doc = json.loads(lines[0])
    assert saved_doc["id"] == "callback-doc"
    assert len(saved_doc["metadata"]["inference_results"]) == len(parts)


def test_query_builder_none_payload_skips_document(tmp_path):
    output_dir = tmp_path / "none_payload_output"
    documents = [Document(text="skip me", id="skip-none")]

    def none_query_builder(runner, document):
        return None

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        temperature=0.0,
        model_max_context=2048,
        max_concurrent_requests=1,
        max_concurrent_tasks=1,
        metric_interval=60,
    )

    runner = InferenceRunner(
        query_builder=none_query_builder,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0, world_size=1))

    doc = documents[0]
    assert not doc.metadata.get("inference_results"), "Document should have no inference results when payload is None"
    output_file = output_dir / "00000.jsonl"
    assert not output_file.exists() or output_file.read_text().strip() == "", (
        "No output should be written for skipped document"
    )


def test_async_query_builder_none_payload_skips_document(tmp_path):
    output_dir = tmp_path / "none_async_output"
    documents = [Document(text="skip me async", id="skip-async")]

    async def none_async_builder(runner, document):
        yield None

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        temperature=0.0,
        model_max_context=2048,
        max_concurrent_requests=1,
        max_concurrent_tasks=1,
        metric_interval=60,
    )

    runner = InferenceRunner(
        query_builder=none_async_builder,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0, world_size=1))

    doc = documents[0]
    assert not doc.metadata.get("inference_results"), (
        "Document should have no inference results when async payload is None"
    )
    output_file = output_dir / "00000.jsonl"
    assert not output_file.exists() or output_file.read_text().strip() == "", (
        "No output should be written for skipped document"
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
        logs_path = Path(temp_dir) / "logs"

        # Create test documents
        documents = [Document(text=f"Test document {i}", id=str(i)) for i in range(num_docs)]

        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="test-model",
            temperature=0.0,
            model_max_context=8192,
            max_concurrent_requests=2,  # Low concurrency to ensure predictable chunk completion
            max_concurrent_tasks=2,
            metric_interval=120,
        )

        # === FIRST RUN: Should fail partway through ===
        failing_query_builder = ControlledQueryBuilder(fail_after_count=fail_after_docs)

        pipeline_executor_1 = LocalPipelineExecutor(
            pipeline=[
                documents,
                InferenceRunner(
                    query_builder=failing_query_builder,
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
            logging_dir=str(logs_path / "checkpoint_test_run1"),
            tasks=1,
        )

        # Run first pipeline - should fail due to query_builder exception
        try:
            pipeline_executor_1.run()
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
        success_query_builder = ControlledQueryBuilder()  # No failures this time

        pipeline_executor_2 = LocalPipelineExecutor(
            pipeline=[
                documents,  # Same document list
                InferenceRunner(
                    query_builder=success_query_builder,
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
            logging_dir=str(logs_path / "checkpoint_test_run2"),
            tasks=1,
        )

        # Run second pipeline - should complete successfully
        pipeline_executor_2.run()

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
            assert "inference_results" in doc["metadata"], f"Document {doc['id']} missing inference_results"

            inference_results = doc["metadata"]["inference_results"]
            assert len(inference_results) > 0, f"Document {doc['id']} has no inference results"

            # Verify inference result structure (dummy server should return success)
            for result in inference_results:
                assert "text" in result, f"Inference result missing 'text' field for doc {doc['id']}"
                assert "finish_reason" in result, f"Inference result missing 'finish_reason' field for doc {doc['id']}"
                assert "usage" in result, f"Inference result missing 'usage' field for doc {doc['id']}"


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
        def normal_query_builder(runner, document):
            return {
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

        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="reducto/RolmOCR",
            temperature=0.0,
            model_max_context=8192,
            max_concurrent_requests=500,
            max_concurrent_tasks=500,
            metric_interval=120,
        )

        pipeline_executor = LocalPipelineExecutor(
            pipeline=[
                documents,
                InferenceRunner(
                    query_builder=normal_query_builder,
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
            inference_results = doc["metadata"]["inference_results"]
            assert len(inference_results) > 0, f"Document {doc['id']} has no inference results"

            # All results should be successful (dummy server always succeeds)
            for result in inference_results:
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
