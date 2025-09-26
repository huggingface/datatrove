"""
Inference pipeline for running LLM inference on documents.

This module provides infrastructure for running inference on documents using various
inference servers like SGLang and VLLM. It supports concurrent processing, metrics
collection, and post-processing steps.

Parts of this implementation are adapted from https://github.com/allenai/olmocr
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Iterable, Literal, Sequence

from loguru import logger

from datatrove.data import Document
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.servers import (
    DummyServer,
    InferenceServer,
    SGLangServer,
    VLLMServer,
)
from datatrove.pipeline.inference.utils.metrics import MetricsKeeper, QueueSizesKeeper
from datatrove.pipeline.writers.disk_base import DiskWriter


@dataclass
class InferenceSuccess:
    """
    Successful inference result.
    
    Attributes:
        text: Generated text from the model
        finish_reason: Reason why generation finished
        usage: Token usage statistics from the model
    """
    text: str
    finish_reason: str
    usage: dict


@dataclass
class InferenceError:
    """
    Failed inference result.
    
    Attributes:
        error: Error message describing what went wrong
    """
    error: str


class InferenceProcessingError(Exception):
    """
    Exception raised when document inference processing fails.
    
    Attributes:
        document: The original document that failed processing
        error: The underlying error that caused the failure
    """
    def __init__(self, document: Document, error: str | Exception):
        self.document = document
        self.error = error
        super().__init__(f"Failed to process document {document.id}: {error}")


# --------------------------------------------------------------------------- #
# Low-level, dependency-free HTTP POST helper (kept from the original file)
# --------------------------------------------------------------------------- #
async def _raw_post(url: str, json_data: dict) -> tuple[int, bytes]:
    """
    Very small HTTP/1.1 POST helper using the std-lib socket machinery.
    
    Args:
        url: The target URL for the POST request
        json_data: Dictionary to be sent as JSON payload
        
    Returns:
        Tuple of (status_code, response_body)
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port or 80
    path = parsed.path or "/"

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    reader, writer = await asyncio.open_connection(host, port)
    try:
        payload = json.dumps(json_data).encode()
        request = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode()
        writer.write(request + payload)
        await writer.drain()

        # Status line
        status_parts = (await reader.readline()).decode().split(" ", 2)
        status_code = int(status_parts[1]) if len(status_parts) >= 2 else 500

        # Headers (ignored – we rely on Content-Length only)
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break

        # Body
        body = await reader.read()  # connection closes -> EOF
        return status_code, body
    finally:
        writer.close()
        await writer.wait_closed()


# --------------------------------------------------------------------------- #
# Public, simplified configuration
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    """
    Configuration for inference server and processing parameters.
    
    Attributes:
        server_type: Type of inference server to use
        model_name_or_path: Path or name of the model to load
        temperature: Sampling temperature for generation
        model_max_context: Maximum context length for the model
        max_concurrent_requests: Maximum number of concurrent requests to server
        max_concurrent_tasks: Maximum number of concurrent processing tasks 
            If your query_builder is slow, it's better to provide higher value than concurrent requests
            to ensure that there are always enough requests to keep the server busy
        metric_interval: Interval for metrics reporting in seconds
        records_per_chunk: Number of records per processing chunk (None for no chunking)
            This is useful for scenarios where the job can be killed at any time and you don't want to lose all the progress
        model_kwargs: Additional keyword arguments for model initialization (Will be provided as --key=value to the model)
    """
    server_type: Literal["sglang", "vllm", "dummy"]
    model_name_or_path: str
    temperature: float = 0.0
    model_max_context: int = 8192
    max_concurrent_requests: int = 500
    max_concurrent_tasks: int = 500
    metric_interval: int = 120
    records_per_chunk: int | None = None
    model_kwargs: dict | None = None

# --------------------------------------------------------------------------- #
# Minimal inference runner
# --------------------------------------------------------------------------- #
class InferenceRunner(PipelineStep):
    """
    Pipeline step for running inference on documents using various inference servers.
    
    This runner pulls documents from readers, converts them to LLM requests via a query builder,
    sends requests to a locally spawned inference server, and processes the responses through
    post-processing steps.

    Inference results are saved in document metadata as "inference_results" list.
    Each inference result is either InferenceSuccess or InferenceError.
    """
    name = "Inference 🔍"
    type = "Model call"

    def __init__(
        self,
        query_builder: Callable[[InferenceRunner, Document], AsyncGenerator[dict, None] | dict],
        config: InferenceConfig,
        post_process_steps: PipelineStep | Sequence[PipelineStep],
        completions_dir: DataFolderLike | None = None,
        exclusion_writer: DiskWriter | None = None,
    ):
        """
        Initialize the inference runner.
        
        Args:
            query_builder: Function that returns inference request payload(s) for a document.
                          Can return either:
                          - AsyncGenerator[dict, None]: async generator yielding dicts
                          - dict: single payload dict
            config: Configuration for the inference server and processing
            post_process_steps: Pipeline steps to run after each document is processed
            completions_dir: Directory for storing checkpoints (only relevant if records_per_chunk is provided)
            exclusion_writer: Optional writer for saving failed documents
        """
        super().__init__()

        # Normalize post_process_steps to a list
        if isinstance(post_process_steps, Sequence) and not isinstance(post_process_steps, (str, bytes)):
            self.post_process_steps = list(post_process_steps)
        else:
            self.post_process_steps = [post_process_steps]

        self.query_builder = query_builder
        self.config = config
        self.completions_dir = (
            get_datafolder(completions_dir) if completions_dir else None
        )
        self.exclusion_writer = exclusion_writer
        self._server: InferenceServer | None = None
        self.metrics = MetricsKeeper(window=60*5)
        self.queue_sizes = QueueSizesKeeper()

    async def metrics_reporter(self, interval: int = 600):
        """
        Periodically report metrics and queue sizes.
        
        Args:
            interval: Reporting interval in seconds
        """
        while True:
            # Leading newlines preserve table formatting in logs
            logger.info("\n" + str(self.metrics))
            logger.info("\n" + str(self.queue_sizes))
            logger.info(str(self.stats))
            await asyncio.sleep(interval)

    @property
    def server(self) -> InferenceServer:
        """
        Lazy initialization of the inference server.
        
        Returns:
            The initialized inference server instance
        """
        if self._server is None:
            self._server = self._init_server()
        # At this point _server is guaranteed to be not None after _init_server()
        assert self._server is not None
        return self._server

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _init_server(self) -> InferenceServer:
        """
        Spawn the requested inference server (non-blocking).
        
        Returns:
            The initialized inference server instance
            
        Raises:
            ValueError: If unsupported server type is specified
        """
        stype = self.config.server_type
        
        if stype == "sglang":
            return SGLangServer(
                self.config.model_name_or_path,
                self.config.model_max_context,
                self.config.model_kwargs,
            )
        elif stype == "vllm":
            return VLLMServer(
                self.config.model_name_or_path,
                self.config.model_max_context,
                self.config.model_kwargs,
            )
        elif stype == "dummy":
            # Dummy server only uses standard library modules
            return DummyServer(
                self.config.model_name_or_path,
                self.config.model_kwargs,
            )
        else:
            raise ValueError(f"Unsupported server type: {stype}")

    async def _send_request(self, payload: dict, semaphore: asyncio.Semaphore) -> InferenceSuccess | InferenceError:
        """
        POST payload to the local server and return the parsed result.
        
        Args:
            payload: The request payload to send
            semaphore: Semaphore for controlling concurrent requests
            
        Returns:
            InferenceSuccess with response data or InferenceError with error message
        """
        url = f"http://localhost:{self.server.port}/v1/chat/completions"
        max_retries = 6
        attempt = 0

        self.queue_sizes.change_queues({"waiting_requests": 1})
        async with semaphore:
            self.queue_sizes.change_queues({"waiting_requests": -1})
            self.queue_sizes.change_queues({"running_requests": 1})

            while attempt < max_retries:
                try:
                    status, body = await _raw_post(url, json_data=payload)
                    if status == 400:
                        self.queue_sizes.change_queues({"running_requests": -1})
                        return InferenceError(error=f"Got BadRequestError from server: {body.decode()}, skipping this response")
                    elif status == 500:
                        self.queue_sizes.change_queues({"running_requests": -1})
                        return InferenceError(error=f"Got InternalServerError from server: {body.decode()}, skipping this response")
                    elif status != 200:
                        self.queue_sizes.change_queues({"running_requests": -1})
                        return InferenceError(error=f"Error http status {status}")

                    response = json.loads(body)
                    choice = response["choices"][0]

                    # Track metrics
                    usage = response.get("usage", {})
                    self.metrics.add_metrics(
                        tokens_input=usage.get("prompt_tokens", 0),
                        tokens_output=usage.get("completion_tokens", 0),
                    )

                    self.queue_sizes.change_queues({"running_requests": -1})
                    return InferenceSuccess(
                        text=choice["message"]["content"],
                        finish_reason=choice["finish_reason"],
                        usage=usage
                    )
                except (ConnectionError, OSError, asyncio.TimeoutError) as e:
                    # This means the server is dead likely, so we need to wait for restart
                    logger.warning(f"Client error: {type(e)} {e}")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    sleep_delay = 5 * (2**attempt)
                    await asyncio.sleep(sleep_delay)
                    attempt += 1
                except asyncio.CancelledError:
                    logger.info(f"Request cancelled")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    raise
                except Exception as e:
                    logger.warning(f"Unexpected error: {type(e)} {e}")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    return InferenceError(error=str(e))

            self.queue_sizes.change_queues({"running_requests": -1})
            return InferenceError(error=f"Failed to process request after {max_retries} attempts")

    def _read_checkpoint(self, rank: int) -> tuple[int, int]:
        """
        Read the last completed chunk index from checkpoint file.
        
        Args:
            rank: Process rank identifier
            
        Returns:
            Tuple of (last_completed_chunk_index, total_documents_processed)
        """
        if self.completions_dir is None or self.config.records_per_chunk is None:
            return -1, 0

        checkpoint_file = f"{rank}.txt"
        if self.completions_dir.exists(checkpoint_file):
            content = str(self.completions_dir.read_text(checkpoint_file)).strip()
            if content:
                lines = content.split("\n")
                if len(lines) >= 2:
                    return int(lines[0]), int(lines[1])
                elif len(lines) == 1:
                    return int(lines[0]), 0
        return -1, 0

    def _write_checkpoint(self, rank: int, chunk_index: int, total_documents_processed: int):
        """
        Write the completed chunk index to checkpoint file.
        
        Args:
            rank: Process rank identifier
            chunk_index: Index of the completed chunk
            total_documents_processed: Total number of documents processed so far
        """
        if self.completions_dir is None:
            return

        checkpoint_file = f"{rank}.txt"
        content = f"{chunk_index}\n{total_documents_processed}"
        self.completions_dir.write_text(checkpoint_file, content)

    async def _exhaust_task_pool(self, tasks_pool: set, rank: int, world_size: int, chunk_index: int | None = None, exclusion_writer_context=None) -> int:
        """
        Exhaust all remaining tasks in the pool and return count of processed documents.
        
        Args:
            tasks_pool: Set of asyncio tasks to complete
            rank: Process rank identifier
            world_size: Total number of processes in distributed setup
            chunk_index: Optional chunk index for document metadata
            exclusion_writer_context: Context manager for exclusion writer
            
        Returns:
            Number of documents successfully processed
        """
        documents_processed = 0
        while tasks_pool:
            done, tasks_pool = await asyncio.wait(tasks_pool, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    result_document = task.result()
                    await self._save_document(result_document, rank, world_size, chunk_index if self.config.records_per_chunk else None)
                    documents_processed += 1
                except InferenceProcessingError as e:
                    logger.warning(f"Document processing failed: {e}")
                    self.stat_update("failed_documents", value=1, unit="documents")
                    if self.exclusion_writer and exclusion_writer_context:
                        exclusion_writer_context.write(e.document, rank)
                except Exception as e:
                    logger.exception(f"Unexpected error processing document: {e}")
                    self.stat_update("failed_documents", value=1, unit="documents")
        return documents_processed

    async def _save_document(self, document: Document, rank: int, world_size: int, chunk_index: int | None = None):
        """
        Save processed document through post-processing pipeline.
        
        Args:
            document: The processed document to save
            rank: Process rank identifier
            world_size: Total number of processes in distributed setup
            chunk_index: Optional chunk index to add to document metadata
        """
        # Add chunk_index to document metadata if chunking is enabled
        if chunk_index is not None:
            document.metadata["chunk_index"] = str(chunk_index)

        # Track document metrics
        try:
            inference_results = document.metadata.get("inference_results", [])  # type: ignore
            successful_requests = sum(1 for result in inference_results if isinstance(result, InferenceSuccess))  # type: ignore
            failed_requests = len(inference_results) - successful_requests  # type: ignore

            # Track tokens for each inference result
            total_input_tokens = 0
            total_output_tokens = 0
            for result in inference_results:
                if isinstance(result, InferenceSuccess):
                    prompt_tokens = result.usage.get("prompt_tokens", 0)  # type: ignore
                    completion_tokens = result.usage.get("completion_tokens", 0)  # type: ignore
                    total_input_tokens += prompt_tokens
                    total_output_tokens += completion_tokens
                    
                    # Update stats for each individual request
                    self.stat_update("prompt_tokens", value=prompt_tokens, unit="request")
                    self.stat_update("completion_tokens", value=completion_tokens, unit="request")

            self.metrics.add_metrics(
                tokens_finished_input=total_input_tokens,
                tokens_finished_output=total_output_tokens,
                requests=len(inference_results)  # type: ignore
            )

            self.stat_update("successful_requests", value=successful_requests, unit="document")
            self.stat_update("failed_requests", value=failed_requests, unit="document")
            self.stat_update("successful_documents", value=1)

        except Exception as e:
            logger.warning(f"Failed to process inference results for metrics: {e}")
            self.stat_update("failed_documents", value=1)

        # Run through post-processing pipeline
        tmp_gen = (d for d in [document])
        for step in self.post_process_steps:
            tmp_gen = step.run(tmp_gen, rank, world_size=world_size)

        # Exhaust the generator to trigger all post-processing steps
        deque(tmp_gen, maxlen=0)

    async def _async_data_gen(self, sync_gen: Iterable[Document]):
        """
        Convert synchronous generator to async generator using asyncio.to_thread.
        
        Args:
            sync_gen: Synchronous iterable of documents
            
        Yields:
            Document objects from the synchronous generator
        """
        def get_next_item(iterator):
            try:
                return next(iterator), False
            except StopIteration:
                return None, True

        iterator = iter(sync_gen)
        while True:
            item, is_done = await asyncio.to_thread(get_next_item, iterator)
            if is_done:
                break
            yield item

    # --------------------------------------------------------------------- #
    # Async processing
    # --------------------------------------------------------------------- #
    async def run_async(
        self,
        data_gen: Iterable[Document],
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Run asynchronous inference processing on the provided data.
        
        Args:
            data_gen: Iterable of Document objects to process
            rank: Process rank identifier for distributed processing
            world_size: Total number of processes in distributed setup
        """
        # 1. start server
        self._init_server()
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        server_task = asyncio.create_task(
            self.server.host_server(offset=rank)
        )
        await self.server.wait_until_ready()
        logger.info(f"Inference server up on port {self.server.port}")

        # Start metrics reporting
        self.metrics.reset()
        metrics_task = asyncio.create_task(self.metrics_reporter(interval=self.config.metric_interval))

        # 2. Initialize processing state
        last_completed_chunk, total_documents_processed = self._read_checkpoint(rank)
        chunk_index = last_completed_chunk + 1
        documents_to_skip = total_documents_processed

        if documents_to_skip > 0:
            logger.info(f"Resuming from chunk {chunk_index}, will skip {documents_to_skip} already processed documents")

        # 3. Processing state variables
        tasks_pool: set[asyncio.Task] = set()
        documents_skipped = 0
        chunk_documents_read = 0

        async def _handle_record(doc: Document) -> Document:
            """
            Handle inference requests for a single document.
            
            Args:
                doc: Document to process
                
            Returns:
                Document with inference results in metadata
                
            Raises:
                InferenceProcessingError: If document processing fails
            """
            try:
                # Get payloads from query_builder
                payloads_result = self.query_builder(self, doc)

                # Handle different return types
                request_tasks = []

                # Check if it's an async generator
                if isinstance(payloads_result, AsyncGenerator):
                    # It's an async generator - process each payload as soon as it's yielded
                    async for payload in payloads_result:
                        # Set default values for payload
                        payload.setdefault("model", self.config.model_name_or_path)
                        payload.setdefault("temperature", self.config.temperature)

                        # Start request immediately
                        task = asyncio.create_task(self._send_request(payload, semaphore))
                        request_tasks.append(task)

                elif isinstance(payloads_result, dict):
                    # Single dict
                    payload = payloads_result
                    payload.setdefault("model", self.config.model_name_or_path)
                    payload.setdefault("temperature", self.config.temperature)
                    task = asyncio.create_task(self._send_request(payload, semaphore))
                    request_tasks.append(task)

                if not request_tasks:
                    raise InferenceProcessingError(doc, "No valid payloads generated from query_builder")

                # Wait for all requests to complete and collect results in order
                results = await asyncio.gather(*request_tasks)

                # Store results directly in document metadata
                doc.metadata["inference_results"] = results  # type: ignore

                return doc
            except InferenceProcessingError:
                # Re-raise InferenceProcessingError as-is
                raise
            except Exception as e:
                # Wrap other exceptions in InferenceProcessingError
                raise InferenceProcessingError(doc, e)

        # 4. Main processing loop - unified for both chunked and non-chunked, now async
        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as exclusion_writer_context:
            async for record in self._async_data_gen(data_gen):
                # Skip documents if resuming from checkpoint
                if documents_skipped < documents_to_skip:
                    documents_skipped += 1
                    continue

                # Throttle by task pool size
                while len(tasks_pool) >= self.config.max_concurrent_tasks:
                    done, tasks_pool = await asyncio.wait(tasks_pool, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        try:
                            result_document = task.result()
                            await self._save_document(result_document, rank, world_size, chunk_index if self.config.records_per_chunk else None)
                        except InferenceProcessingError as e:
                            logger.warning(f"Document processing failed: {e}")
                            self.stat_update("failed_documents", value=1, unit="documents")
                            if exclusion_writer_context:
                                exclusion_writer_context.write(e.document, rank)
                        except Exception as e:
                            logger.exception(f"Unexpected error processing document: {e}")
                            self.stat_update("failed_documents", value=1, unit="documents")

                # Add task for current record
                tasks_pool.add(asyncio.create_task(_handle_record(record)))  # type: ignore

                # Update counters
                if self.config.records_per_chunk is not None:
                    chunk_documents_read += 1
                    total_documents_processed += 1

                    # Check if chunk is complete
                    if chunk_documents_read >= self.config.records_per_chunk:
                        # Exhaust all remaining tasks for this chunk
                        await self._exhaust_task_pool(tasks_pool, rank, world_size, chunk_index, exclusion_writer_context)
                        tasks_pool = set()

                        # Update checkpoint and prepare for next chunk
                        self._write_checkpoint(rank, chunk_index, total_documents_processed)
                        logger.info(f"Completed chunk {chunk_index}, processed {self.config.records_per_chunk} documents")

                        chunk_documents_read = 0
                        chunk_index += 1

            # 5. Process any remaining tasks
            await self._exhaust_task_pool(tasks_pool, rank, world_size, chunk_index, exclusion_writer_context)

        # 6. shutdown inference server and metrics
        server_task.cancel()
        metrics_task.cancel()

    # --------------------------------------------------------------------- #
    # Synchronous entrypoint required by PipelineStep
    # --------------------------------------------------------------------- #
    def run(
        self,
        data: Iterable[Document],
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Consume `data`, run inference and post-processing, do not yield further documents.
        
        Args:
            data: Iterable of Document objects to process
            rank: Process rank identifier for distributed processing
            world_size: Total number of processes in distributed setup
        """
        with self.track_time():
            asyncio.run(self.run_async(data, rank, world_size))
