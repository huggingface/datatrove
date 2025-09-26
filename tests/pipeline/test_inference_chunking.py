"""
Tests for inference chunking functionality.

Tests the chunking behavior of the InferenceRunner end-to-end with DummyServer,
including checkpoint management, document processing in chunks, and recovery mechanisms.
"""

import os
import tempfile
import pytest

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.run_inference import (
    InferenceConfig,
    InferenceRunner,
    InferenceSuccess,
)


class CollectorStep(PipelineStep):
    """
    Collect documents that flow through this step for verification.
    
    Used to capture processed documents from the inference pipeline
    for verification in tests.
    """
    
    def __init__(self):
        super().__init__()
        self.collected: list[Document] = []

    def run(self, data, rank: int = 0, world_size: int = 1):
        """
        Collect documents passing through the pipeline.
        
        Args:
            data: Iterable of Document objects
            rank: Process rank for distributed processing
            world_size: Total number of processes
            
        Yields:
            Document: Each processed document
        """
        for doc in data:
            self.collected.append(doc)
            yield doc


def create_test_documents(count: int) -> list[Document]:
    """
    Create test documents for testing.
    
    Args:
        count: Number of documents to create
        
    Returns:
        List of test Document objects
    """
    documents = []
    for i in range(count):
        doc = Document(
            text=f"Test document {i}",
            id=f"doc_{i}",
            metadata={"original_index": i}
        )
        documents.append(doc)
    return documents


def simple_query_builder(doc: Document) -> dict:
    """
    Simple query builder for testing.
    
    Args:
        doc: Document to process
        
    Returns:
        OpenAI-compatible request payload
    """
    return {
        "messages": [{"role": "user", "content": doc.text}],
        "max_tokens": 100
    }


def test_chunking_with_chunk_index():
    """
    Test end-to-end inference with chunking to verify chunk_index is properly set.
    
    Verifies that:
    1. Documents are processed in chunks with the DummyServer
    2. Each document gets chunk_index metadata correctly set
    3. Chunk indices match the expected chunking pattern
    """
    # Create 7 documents to test chunking with chunk size 3
    docs = create_test_documents(7)
    collector = CollectorStep()
    
    config = InferenceConfig(
        server_type="dummy",
        records_per_chunk=3,  # Process 3 documents per chunk
        model_name_or_path="test-model",
        temperature=0.0,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = InferenceRunner(
            query_builder=simple_query_builder,
            config=config,
            post_process_steps=collector,
            completions_dir=tmpdir,
        )
        
        # Run inference end-to-end with DummyServer
        runner.run(docs)
        
        # Verify all documents were processed
        assert len(collector.collected) == len(docs)
        
        # Verify chunk_index metadata is set correctly
        expected_chunk_mapping = {
            "doc_0": "0", "doc_1": "0", "doc_2": "0",  # Chunk 0: first 3 docs
            "doc_3": "1", "doc_4": "1", "doc_5": "1",  # Chunk 1: next 3 docs  
            "doc_6": "2",                              # Chunk 2: last 1 doc
        }
        
        for doc in collector.collected:
            # Verify inference results are present
            assert "inference_results" in doc.metadata
            inference_results = doc.metadata["inference_results"]
            assert len(inference_results) == 1
            assert isinstance(inference_results[0], InferenceSuccess)
            
            # Verify chunk_index is correctly set
            assert "chunk_index" in doc.metadata
            expected_chunk = expected_chunk_mapping[doc.id]
            assert doc.metadata["chunk_index"] == expected_chunk
        
        # Verify checkpoint files were created
        checkpoint_files = [f for f in os.listdir(tmpdir) if f.endswith('.txt')]
        assert len(checkpoint_files) == 1  # Should have one checkpoint file (0.txt)
        
        # Verify checkpoint content
        with open(os.path.join(tmpdir, "0.txt"), 'r') as f:
            content = f.read().strip()
            lines = content.split('\n')
            assert len(lines) == 2
            last_chunk_index = int(lines[0])
            total_documents = int(lines[1])
            # We don't update the checkpoint all docs are processed as there is no point
            assert last_chunk_index == 1
            assert total_documents == 6


@pytest.mark.timeout(60)
def test_checkpoint_recovery_skips_documents():
    """
    Test that checkpoint recovery properly skips already processed documents.
    
    Verifies that:
    1. First run processes some documents and creates checkpoints
    2. Second run with same completions_dir skips already processed documents
    3. Only new documents are processed in the second run
    """
    # Create documents for testing
    initial_docs = create_test_documents(4)  # First batch: doc_0 to doc_3
    additional_docs = initial_docs + create_test_documents(6)[4:]  # Original docs + doc_4, doc_5
    
    config = InferenceConfig(
        server_type="dummy",
        records_per_chunk=2,  # Process 2 documents per chunk
        model_name_or_path="test-model",
        temperature=0.0,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # First run: process initial documents
        collector1 = CollectorStep()
        runner1 = InferenceRunner(
            query_builder=simple_query_builder,
            config=config,
            post_process_steps=collector1,
            completions_dir=tmpdir,
        )
        
        runner1.run(initial_docs)
        
        # Verify first run processed all documents
        assert len(collector1.collected) == 4
        processed_ids_first = [doc.id for doc in collector1.collected]
        assert processed_ids_first == ["doc_0", "doc_1", "doc_2", "doc_3"]
        
        # Verify checkpoint was created
        checkpoint_files = os.listdir(tmpdir)
        assert "0.txt" in checkpoint_files
        
        # Read checkpoint to verify state
        with open(os.path.join(tmpdir, "0.txt"), 'r') as f:
            content = f.read().strip()
            lines = content.split('\n')
            last_chunk_index = int(lines[0])
            total_documents = int(lines[1])
            # We don't update the checkpoint all docs are processed as there is no point
            assert last_chunk_index == 1  # Chunks 0 and 1 completed
            assert total_documents == 4   # 4 documents processed
        
        # Second run: process additional documents (original + new documents)
        collector2 = CollectorStep()
        runner2 = InferenceRunner(
            query_builder=simple_query_builder,
            config=config,
            post_process_steps=collector2,
            completions_dir=tmpdir,  # Same completions directory
        )
        
        runner2.run(additional_docs)
        
        # Verify second run only processed new documents (doc_4 and doc_5)
        # Documents doc_0, doc_1, doc_2, doc_3 should have been skipped
        assert len(collector2.collected) == 2
        processed_ids_second = [doc.id for doc in collector2.collected]
        assert processed_ids_second == ["doc_4", "doc_5"]
        
        # Verify the new documents have correct chunk indices
        # They should continue from where the first run left off
        for doc in collector2.collected:
            assert "chunk_index" in doc.metadata
            if doc.id == "doc_4":
                assert doc.metadata["chunk_index"] == "2"  # Next chunk after checkpoint
            elif doc.id == "doc_5":
                assert doc.metadata["chunk_index"] == "2"  # Same chunk as doc_4


@pytest.mark.timeout(60)
def test_chunks_properly_saved():
    """
    Test that chunks are properly saved with correct checkpoint management.
    
    Verifies that:
    1. Each chunk completion updates the checkpoint file
    2. Checkpoint contains correct chunk index and document count
    3. Documents within chunks are processed together
    """
    # Create 5 documents to test 3 chunks: [2, 2, 1]
    docs = create_test_documents(5)
    collector = CollectorStep()
    
    config = InferenceConfig(
        server_type="dummy",
        records_per_chunk=2,  # Process 2 documents per chunk
        model_name_or_path="test-model",
        temperature=0.0,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = InferenceRunner(
            query_builder=simple_query_builder,
            config=config,
            post_process_steps=collector,
            completions_dir=tmpdir,
        )
        
        # Run inference
        runner.run(docs)
        
        # Verify all documents were processed
        assert len(collector.collected) == 5
        
        # Verify chunk distribution
        chunk_distribution = {}
        for doc in collector.collected:
            chunk_idx = doc.metadata["chunk_index"]
            if chunk_idx not in chunk_distribution:
                chunk_distribution[chunk_idx] = []
            chunk_distribution[chunk_idx].append(doc.id)
        
        # Should have 3 chunks: 0, 1, 2
        assert len(chunk_distribution) == 3
        assert "0" in chunk_distribution
        assert "1" in chunk_distribution  
        assert "2" in chunk_distribution
        
        # Verify chunk sizes
        assert len(chunk_distribution["0"]) == 2  # First chunk: 2 documents
        assert len(chunk_distribution["1"]) == 2  # Second chunk: 2 documents
        assert len(chunk_distribution["2"]) == 1  # Third chunk: 1 document
        
        # Verify document order within chunks
        assert chunk_distribution["0"] == ["doc_0", "doc_1"]
        assert chunk_distribution["1"] == ["doc_2", "doc_3"]
        assert chunk_distribution["2"] == ["doc_4"]
        
        # Verify final checkpoint
        checkpoint_file = os.path.join(tmpdir, "0.txt")
        assert os.path.exists(checkpoint_file)
        
        with open(checkpoint_file, 'r') as f:
            content = f.read().strip()
            lines = content.split('\n')
            assert len(lines) == 2
            last_chunk_index = int(lines[0])
            total_documents = int(lines[1])
            
            # Should have completed chunk 2 (the last chunk)
            # We don't update the checkpoint all docs are processed as there is no point
            assert last_chunk_index == 1
            assert total_documents == 4
        
        # Verify all documents have inference results
        for doc in collector.collected:
            assert "inference_results" in doc.metadata
            inference_results = doc.metadata["inference_results"]
            assert len(inference_results) == 1
            assert isinstance(inference_results[0], InferenceSuccess)


def test_chunking_algorithm_logic():
    """
    Test the chunking algorithm logic without server dependencies.
    
    Verifies that documents are properly divided into chunks of the
    specified size, with correct handling of partial final chunks.
    """
    documents = create_test_documents(10)
    
    # Test chunking with records_per_chunk=3
    records_per_chunk = 3
    chunk_index = 0
    chunk_documents_read = 0
    chunks = []
    current_chunk = []
    
    for doc in documents:
        current_chunk.append(doc)
        chunk_documents_read += 1
        
        if chunk_documents_read >= records_per_chunk:
            chunks.append((chunk_index, current_chunk.copy()))
            current_chunk = []
            chunk_documents_read = 0
            chunk_index += 1
    
    # Add remaining documents as the last chunk
    if current_chunk:
        chunks.append((chunk_index, current_chunk))
    
    # Verify chunking results
    assert len(chunks) == 4  # 3 full chunks + 1 partial chunk
    assert len(chunks[0][1]) == 3  # First chunk has 3 documents
    assert len(chunks[1][1]) == 3  # Second chunk has 3 documents  
    assert len(chunks[2][1]) == 3  # Third chunk has 3 documents
    assert len(chunks[3][1]) == 1  # Last chunk has 1 document
    
    # Verify document IDs are preserved
    all_docs_from_chunks = []
    for chunk_idx, chunk_docs in chunks:
        for doc in chunk_docs:
            all_docs_from_chunks.append(doc.id)
    
    expected_ids = [f"doc_{i}" for i in range(10)]
    assert all_docs_from_chunks == expected_ids 