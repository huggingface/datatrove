"""
Tests for the InferenceRunner pipeline step.

Tests the end-to-end functionality of running inference on documents
using various server backends and processing configurations.
"""

import asyncio
import json
import os
import random
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import (
    InferenceRunner,
    InferenceConfig,
    InferenceSuccess,
    InferenceError,
)
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter


class CollectorStep(PipelineStep):
    """
    Collect documents that flow through this step for assertions.
    
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
            Document: Each document after collecting it
        """
        for doc in data:
            self.collected.append(doc)
            # Propagate so the pipeline can keep going if needed
            yield doc


class ExclusionCollectorWriter(JsonlWriter):
    """
    Custom writer for testing exclusion functionality.
    
    Collects excluded documents for verification in tests.
    """
    
    def __init__(self, output_folder: str):
        super().__init__(output_folder)
        self.excluded_docs: list[Document] = []
    
    def write(self, doc: Document, rank: int = 0, *args, **kwargs):
        """
        Override write to collect excluded documents.
        
        Args:
            doc: Document being excluded
            rank: Process rank
        """
        self.excluded_docs.append(doc)


def simple_query_builder(doc: Document) -> dict:
    """
    Simple query builder for testing that creates OpenAI-compatible requests.
    
    Args:
        doc: Document to process
        
    Returns:
        OpenAI-compatible request payload
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Process document: {doc.id}",
            }
        ],
        "max_tokens": 100,
    }


def failing_query_builder(doc: Document) -> dict:
    """
    Query builder that always fails for testing exclusion writer.
    
    Args:
        doc: Document to process
        
    Returns:
        Never returns, always raises exception
        
    Raises:
        RuntimeError: Always raised to simulate query builder failure
    """
    raise RuntimeError(f"Simulated failure processing document {doc.id}")


def test_inference_runner_dummy_end_to_end():
    """
    Test end-to-end inference pipeline with dummy server.
    
    Verifies that:
    - Documents are processed through the inference pipeline
    - Results are properly stored in document metadata
    - Post-processing steps receive the processed documents
    """
    # Prepare test documents
    docs = [
        Document(text="Sample text 1", id="doc1"),
        Document(text="Sample text 2", id="doc2"),
    ]

    # Collect results after inference
    collector = CollectorStep()

    # Configure dummy server for testing
    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        temperature=0.0,
    )

    runner = InferenceRunner(
        query_builder=simple_query_builder,
        config=config,
        post_process_steps=collector,
    )

    # Run the pipeline (synchronously)
    runner.run(docs)

    # Assertions: collector captured as many docs as were supplied
    assert len(collector.collected) == len(docs)
    
    for doc in collector.collected:
        # Verify inference results are present in metadata
        assert "inference_results" in doc.metadata
        
        inference_results = doc.metadata["inference_results"]
        assert isinstance(inference_results, list)
        assert len(inference_results) == 1  # One request per document
        
        result = inference_results[0]
        assert isinstance(result, InferenceSuccess)
        
        # The dummy server returns JSON string with specific structure
        assert result.text.startswith("This is dummy text")
        
        # Verify usage statistics are present
        assert "prompt_tokens" in result.usage
        assert "completion_tokens" in result.usage


def test_inference_runner_async_query_builder():
    """
    Test inference runner with async generator query builder.
    
    Verifies that the runner can handle query builders that return
    async generators yielding multiple requests per document.
    """
    docs = [Document(text="Test", id="doc1")]
    collector = CollectorStep()
    
    async def async_query_builder(doc: Document):
        """Query builder that yields multiple requests asynchronously."""
        # Yield two requests for each document
        yield {
            "messages": [{"role": "user", "content": f"First request for {doc.id}"}],
            "max_tokens": 50,
        }
        yield {
            "messages": [{"role": "user", "content": f"Second request for {doc.id}"}], 
            "max_tokens": 50,
        }

    config = InferenceConfig(server_type="dummy")
    runner = InferenceRunner(
        query_builder=async_query_builder,
        config=config,
        post_process_steps=collector,
    )

    runner.run(docs)
    
    # Should have one processed document
    assert len(collector.collected) == 1
    
    doc = collector.collected[0]
    inference_results = doc.metadata["inference_results"]
    
    # Should have two inference results (one for each yielded request)
    assert len(inference_results) == 2
    assert all(isinstance(result, InferenceSuccess) for result in inference_results)


def test_exclusion_writer_with_query_builder_error():
    """
    Test that exclusion writer captures documents when query builder fails.
    
    Verifies that:
    - Documents with failing query builders are written to exclusion writer
    - Failed documents don't appear in post-processing steps
    - Exclusion writer receives the correct documents
    """
    # Create test documents
    docs = [
        Document(text="Test document 1", id="failing_doc_1"),
        Document(text="Test document 2", id="failing_doc_2"),
    ]
    
    collector = CollectorStep()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create exclusion writer
        exclusion_writer = ExclusionCollectorWriter(tmpdir)
        
        # Configure dummy server with failing query builder
        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="test-model",
            temperature=0.0,
        )
        
        runner = InferenceRunner(
            query_builder=failing_query_builder,  # This will always fail
            config=config,
            post_process_steps=collector,
            exclusion_writer=exclusion_writer,
        )
        
        # Run the pipeline - should handle failures gracefully
        runner.run(docs)
        
        # Verify no documents made it through to post-processing
        assert len(collector.collected) == 0, "No documents should have been processed successfully"
        
        # Verify all documents were written to exclusion writer
        assert len(exclusion_writer.excluded_docs) == len(docs), "All documents should be excluded"
        
        excluded_ids = [doc.id for doc in exclusion_writer.excluded_docs]
        original_ids = [doc.id for doc in docs]
        assert set(excluded_ids) == set(original_ids), "Excluded documents should match original documents"
        

def test_server_auto_restart_behavior():
    """
    Test that verifies the auto-restart behavior works correctly.
    
    Verifies that:
    - Server can be restarted multiple times on failure
    - Max retries limit is respected
    - Port reassignment happens on restart
    """
    from datatrove.pipeline.inference.servers.base import InferenceServer
    
    class MockRestartServer(InferenceServer):
        def __init__(self):
            super().__init__("test-model", "test-template", 8192)
            self.start_count = 0
            self.should_fail_times = 2  # Fail first 2 attempts, succeed on 3rd
            
        async def start_server_task(self, port: int) -> None:
            self.start_count += 1
            self.port = port
            
            if self.start_count <= self.should_fail_times:
                # Simulate server crash/exit
                raise RuntimeError(f"Simulated server failure #{self.start_count}")
            
            # On 3rd attempt, server "stays up" (we'll cancel it manually)
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise
                
        async def is_ready(self) -> bool:
            return self.start_count > self.should_fail_times
    
    async def test_restart_logic():
        server = MockRestartServer()
        
        # Start the server with max_retries=5
        server_task = asyncio.create_task(server.host_server(offset=0, max_retries=5))
        
        # Wait for server to be ready (after restarts)
        await server.wait_until_ready()
        
        # Verify server restarted the expected number of times
        assert server.start_count == 3, f"Expected 3 start attempts, got {server.start_count}"
        assert server.port is not None, "Server should have a port assigned"
        
        # Cancel the server task
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
    
    # Run the async test
    asyncio.run(test_restart_logic())


def test_server_max_retries_exceeded():
    """
    Test that server stops trying after max_retries is exceeded.
    
    Verifies that:
    - Server attempts restart up to max_retries times
    - RuntimeError is raised when max_retries is exceeded
    - Proper error message is included
    """
    from datatrove.pipeline.inference.servers.base import InferenceServer
    
    class FailingServer(InferenceServer):
        def __init__(self):
            super().__init__("test-model", "test-template", 8192)
            self.start_count = 0
            
        async def start_server_task(self, port: int) -> None:
            self.start_count += 1
            self.port = port
            # Always fail
            raise RuntimeError(f"Simulated permanent server failure #{self.start_count}")
                
        async def is_ready(self) -> bool:
            return False
    
    async def test_max_retries():
        server = FailingServer()
        max_retries = 3
        
        # Server should fail after max_retries attempts
        with pytest.raises(RuntimeError, match=f"Failed to start.*after {max_retries} retries"):
            await server.host_server(offset=0, max_retries=max_retries)
        
        # Verify it tried the expected number of times
        assert server.start_count == max_retries, f"Expected {max_retries} attempts, got {server.start_count}"
    
    # Run the async test
    asyncio.run(test_max_retries())
        

