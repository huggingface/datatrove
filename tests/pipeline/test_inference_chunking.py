import asyncio
import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner


class MockPostProcessStep(PipelineStep):
    """Mock post-processing step for testing"""
    def __init__(self):
        super().__init__()
        self.processed_docs = []

    def run(self, data, rank=0, world_size=1):
        for doc in data:
            self.processed_docs.append(doc)
            yield doc


class MockReader(PipelineStep):
    """Mock reader that generates test documents"""
    def __init__(self, documents):
        super().__init__()
        self.documents = documents

    def run(self, data, rank=0, world_size=1):
        for doc in self.documents:
            yield doc


def create_test_documents(count: int) -> list[Document]:
    """Create test documents for testing"""
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
    """Simple query builder for testing"""
    return {
        "messages": [{"role": "user", "content": doc.text}],
        "max_tokens": 100
    }


def test_records_per_chunk_in_config():
    """Test that records_per_chunk can be set in config"""
    config = InferenceConfig(records_per_chunk=5)
    assert config.records_per_chunk == 5
    
    config_none = InferenceConfig()
    assert config_none.records_per_chunk is None


def test_chunking_algorithm_logic():
    """Test the chunking algorithm logic without actual server dependencies"""
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


@pytest.mark.asyncio
async def test_checkpoint_functionality():
    """Test the checkpoint read/write functionality"""
    documents = create_test_documents(5)
    reader = MockReader(documents)
    post_processor = MockPostProcessStep()
    
    config = InferenceConfig(
        records_per_chunk=2,
        server_type="dummy",
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create runner with completions directory
        runner = InferenceRunner(
            records_reader=reader,
            query_builder=simple_query_builder,
            config=config,
            post_process_steps=post_processor,
            completions_dir=tmpdir,
        )
        
        # Test initial checkpoint read (should return -1, 0)
        chunk_index, total_docs = runner._read_checkpoint(rank=0)
        assert chunk_index == -1
        assert total_docs == 0
        
        # Write a checkpoint
        runner._write_checkpoint(rank=0, chunk_index=1, total_documents_processed=4)
        
        # Read the checkpoint back
        chunk_index, total_docs = runner._read_checkpoint(rank=0)
        assert chunk_index == 1
        assert total_docs == 4
        
        # Verify checkpoint file content
        checkpoint_file = os.path.join(tmpdir, "0.txt")
        assert os.path.exists(checkpoint_file)
        with open(checkpoint_file, 'r') as f:
            content = f.read().strip()
            assert content == "1\n4"


def test_chunk_index_in_metadata():
    """Test that chunk_index is properly added to document metadata"""
    doc = Document(text="test", id="test_doc")
    
    # Test _save_document method behavior
    class TestRunner(InferenceRunner):
        def __init__(self):
            # Minimal init without calling super().__init__()
            self.post_process_steps = [MockPostProcessStep()]
    
    runner = TestRunner()
    
    # Test with chunk_index
    asyncio.run(runner._save_document(doc, rank=0, chunk_index=5))
    assert doc.metadata["chunk_index"] == 5
    
    # Test without chunk_index
    doc2 = Document(text="test2", id="test_doc2")
    asyncio.run(runner._save_document(doc2, rank=0, chunk_index=None))
    assert "chunk_index" not in doc2.metadata


@pytest.mark.asyncio
async def test_exhaust_task_pool():
    """Test the _exhaust_task_pool functionality"""
    
    async def mock_task_result(doc_id: str, should_fail: bool = False):
        """Mock task that returns a document or raises an exception"""
        if should_fail:
            raise Exception(f"Failed to process {doc_id}")
        return Document(text=f"processed {doc_id}", id=doc_id)
    
    class TestRunner(InferenceRunner):
        def __init__(self):
            # Minimal init
            self.post_process_steps = [MockPostProcessStep()]
            self.stats = {}
            # Add mock config for the test
            self.config = InferenceConfig(records_per_chunk=5)

        def stat_update(self, *labels, value: int = 1, unit: str | None = None):
            key = "_".join(str(label) for label in labels) if labels else "unknown"
            self.stats[key] = self.stats.get(key, 0) + value

        async def _save_document(self, document, rank, chunk_index=None):
            # Just store the document
            if not hasattr(self, 'saved_docs'):
                self.saved_docs = []
            self.saved_docs.append(document)
    
    runner = TestRunner()
    
    # Create mock tasks
    tasks_pool = {
        asyncio.create_task(mock_task_result("doc1")),
        asyncio.create_task(mock_task_result("doc2")),
        asyncio.create_task(mock_task_result("doc3", should_fail=True)),  # This one will fail
    }
    
    # Exhaust the task pool
    documents_processed = await runner._exhaust_task_pool(tasks_pool, rank=0, chunk_index=1)
    
    # Verify results
    assert documents_processed == 2  # 2 successful documents
    assert len(runner.saved_docs) == 2
    assert runner.stats.get("failed_documents", 0) == 1  # 1 failed document
    
    # Verify saved documents have correct content
    saved_ids = [doc.id for doc in runner.saved_docs]
    assert "doc1" in saved_ids
    assert "doc2" in saved_ids
    assert "doc3" not in saved_ids  # Failed document not saved 