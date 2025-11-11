"""
Inference pipeline for running LLM inference on documents.

This module provides infrastructure for running inference on documents using various
inference servers like SGLang and VLLM. It supports concurrent processing, metrics
collection, and post-processing steps.

Parts of this implementation are adapted from https://github.com/allenai/olmocr
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Awaitable, Callable, Iterable, Literal

from loguru import logger

from datatrove.data import Document
from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.metrics import MetricsKeeper, QueueSizesKeeper
from datatrove.pipeline.inference.servers import (
    DummyServer,
    InferenceServer,
    SGLangServer,
    VLLMServer,
)
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.writers.disk_base import DiskWriter


@dataclass
class InferenceResult:
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


class InferenceError(Exception):
    """
    Exception raised when document inference processing fails.

    Attributes:
        document: The original document that failed processing
        error: The underlying error that caused the failure
    """

    def __init__(self, document: Document, error: str | Exception, payload: dict | None = None):
        self.document = document
        self.error = error
        self.payload = payload
        super().__init__(
            f"Failed to process document {document.id if document is not None else '?'}: {error}. Payload: {payload if payload is not None else '?'}"
        )


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
        server_type: Type of inference server to use.
        model_name_or_path: Path or name of the model to load.
        model_max_context: Maximum context length for the model.
        use_chat: Whether to call the chat endpoint (/v1/chat/completions) instead of the completion endpoint.
        metric_interval: Interval for metrics reporting in seconds.
        tp: Tensor parallelism size (number of GPUs). Converted to the backend-specific flag.
        dp: Data parallelism size (number of model replicas). Converted to the backend-specific flag.
        pp: Pipeline parallelism size (number of pipeline stages). Converted to the backend-specific flag.
        default_generation_params: Default payload parameters for requests sent via the generate callback.
            Callers can override individual keys when building a payload.
        rollouts_per_document: Number of rollouts to perform for each document. Results will be automatically aggregated in the document metadata.
        max_concurrent_generations: Maximum number of generation requests to have in flight at once.
        max_concurrent_documents: Maximum number of documents to process concurrently.
            Defaults to max_concurrent_generations // rollouts_per_document when not provided.
        model_kwargs: Additional keyword arguments passed to model initialization.
        server_log_folder: Optional directory for server logs. Creates one log per rank when set.
    """

    # server and model
    server_type: Literal["sglang", "vllm", "dummy"]
    model_name_or_path: str
    model_max_context: int = 8192
    use_chat: bool = True
    # metrics
    metric_interval: int = 120
    # parallelism
    tp: int = 1
    dp: int = 1
    pp: int = 1
    # rollouts and generation
    default_generation_params: dict = field(default_factory=lambda: {"temperature": 0.0})
    rollouts_per_document: int = 1
    max_concurrent_generations: int = 500
    max_concurrent_documents: int | None = None
    # other
    model_kwargs: dict | None = None
    server_log_folder: str | None = None

    def __post_init__(self):
        if self.max_concurrent_documents is None:
            self.max_concurrent_documents = max(1, self.max_concurrent_generations // self.rollouts_per_document)


# --------------------------------------------------------------------------- #
# Manages output saving, checkpointing, and chunking
# --------------------------------------------------------------------------- #
class CheckpointManager:
    def __init__(self, checkpoints_local_dir: str | None = None, records_per_chunk: int = 6000):
        """
        Manages checkpointing and chunking of documents.
        If checkpoints_local_dir is provided, it will save documents to it in chunks of records_per_chunk documents.
        If it's not provided, it will only write to the main output writer.
        """
        self.checkpoints_local_dir = checkpoints_local_dir if checkpoints_local_dir is not None else None
        self.checkpoints_local_dir_df = (
            get_datafolder(checkpoints_local_dir) if checkpoints_local_dir is not None else None
        )
        if self.checkpoints_local_dir_df is not None and not self.checkpoints_local_dir_df.is_local():
            raise ValueError("checkpoints_local_dir must be a local directory")
        if records_per_chunk <= 0:
            raise ValueError("records_per_chunk must be positive")
        self.records_per_chunk = records_per_chunk

        self.file_locks = defaultdict(asyncio.Lock)
        self.checkpoint_file_lock = asyncio.Lock()
        self.per_chunk_counts = Counter()
        self.new_completed_chunks = set()
        self.last_chunk_index = -1

    async def write_document(self, document: Document, rank: int, chunk_index: int, output_writer_context: DiskWriter):
        """
        Write a document to the checkpoint and main output writer. Potentially closes the main file if the chunk is complete.
        """
        import aiofiles
        import orjson

        should_update_last_chunk_index = False
        async with self.file_locks[chunk_index]:
            # write to main output writer
            if "__no_rollouts_remove" not in document.metadata:
                output_writer_context.write(document, rank=rank, chunk_index=chunk_index)
            self.per_chunk_counts[chunk_index] += 1

            if self.checkpoints_local_dir is not None:
                # save to checkpoint/chunk
                save_path = os.path.join(self.checkpoints_local_dir, f"{rank:05d}/chunk_{chunk_index:05d}.jsonl")
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                if not os.path.exists(save_path):
                    logger.info(f"Creating checkpoint file {save_path}")
                async with aiofiles.open(save_path, "ab") as f:
                    await f.write(orjson.dumps(dataclasses.asdict(document), option=orjson.OPT_APPEND_NEWLINE))
                # see if we have to close the file
                if self.per_chunk_counts[chunk_index] == self.records_per_chunk:
                    # we gotta close the main file
                    output_writer_context.output_mg.pop(
                        output_writer_context._get_output_filename(document, rank, chunk_index=chunk_index)
                    ).close()
                    self.new_completed_chunks.add(chunk_index)
                    should_update_last_chunk_index = True
        # can not be within the chunk lock
        if should_update_last_chunk_index:
            await self.update_last_chunk_index(rank)

    async def parse_existing_checkpoints(self, rank: int, output_writer_context: DiskWriter) -> tuple[int, set[str]]:
        """
        Load all checkpoints for a given rank and write them to the output writer.
        Returns:
        - documents to skip: number of documents from completed chunks that were already finished
        - set of ids of documents that were already processed in the unfinished chunks
        """
        all_ids = set()
        if not self.checkpoints_local_dir:
            return 0, all_ids

        should_update_last_chunk_index = False

        async with self.checkpoint_file_lock:
            if self.checkpoints_local_dir_df.exists(f"last_chunk/{rank:05d}.txt"):
                with self.checkpoints_local_dir_df.open(f"last_chunk/{rank:05d}.txt", "r") as f:
                    self.last_chunk_index = int(f.read().strip())

            reader = JsonlReader(self.checkpoints_local_dir, compression=None)
            # find existing chunk files and read from them
            for filename in self.checkpoints_local_dir_df.glob(f"{rank:05d}/*.jsonl"):
                chunk_index = int(filename.removeprefix(f"{rank:05d}/chunk_").removesuffix(".jsonl"))
                # not strictly needed but just to be safe for the future
                async with self.file_locks[chunk_index]:
                    for document in reader.read_file(filename):
                        if "__no_rollouts_remove" not in document.metadata:
                            output_writer_context.write(document, rank=rank, chunk_index=chunk_index)
                        all_ids.add(document.id)
                        self.per_chunk_counts[chunk_index] += 1
                        if self.per_chunk_counts[chunk_index] == self.records_per_chunk:
                            # close the file
                            output_writer_context.output_mg.pop(
                                output_writer_context._get_output_filename(document, rank, chunk_index=chunk_index)
                            ).close()
                            self.new_completed_chunks.add(chunk_index)
                            # update the last chunk index/delete local file etc
                            should_update_last_chunk_index = True
        # can not be within the chunk lock
        if should_update_last_chunk_index:
            await self.update_last_chunk_index(rank)
        return (self.last_chunk_index + 1) * self.records_per_chunk if self.last_chunk_index >= 0 else 0, all_ids

    async def cleanup_last_chunk(self, rank: int, chunk_index: int):
        import shutil

        if self.checkpoints_local_dir is not None:
            self.new_completed_chunks.add(chunk_index)
            await self.update_last_chunk_index(rank)
            rank_dir = os.path.join(self.checkpoints_local_dir, f"{rank:05d}")
            # second part should be redundant as we technically only call this after everything completes but seems buggy for now
            if os.path.exists(rank_dir) and self.last_chunk_index == chunk_index:
                shutil.rmtree(rank_dir)

    async def update_last_chunk_index(self, rank: int):
        """
        Update the last chunk index and delete the local file if it's complete.
        """
        import os

        async with self.checkpoint_file_lock:
            # possibly multiple ones, in case file +2 finished before +1
            while self.last_chunk_index + 1 in self.new_completed_chunks:
                self.last_chunk_index += 1
                async with self.file_locks[self.last_chunk_index]:
                    chunk_file = os.path.join(
                        self.checkpoints_local_dir, f"{rank:05d}/chunk_{self.last_chunk_index:05d}.jsonl"
                    )
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                logger.info(f"Finished chunk {self.last_chunk_index}")
                # clean up
                self.file_locks.pop(self.last_chunk_index)
                self.per_chunk_counts.pop(self.last_chunk_index)
                self.new_completed_chunks.remove(self.last_chunk_index)
                # save new last chunk index
                with self.checkpoints_local_dir_df.open(f"last_chunk/{rank:05d}.txt", "wt") as f:
                    f.write(str(self.last_chunk_index))

    def chunk_index_gen(self):
        ci = 0
        while True:
            for _ in range(self.records_per_chunk):
                yield ci
            ci += 1


# --------------------------------------------------------------------------- #
# Minimal inference runner
# --------------------------------------------------------------------------- #
class InferenceRunner(PipelineStep):
    """
    Pipeline step for running rollouts on documents using inference servers.

    The runner pulls documents from readers, hands them to a user-provided rollout function,
    and issues generation requests through a locally spawned inference server.
    Rollout outputs are written to document metadata under the configured metadata key.
    """

    name = "Inference 🔍"
    type = "Model call"

    def __init__(
        self,
        rollout_fn: Callable[
            [InferenceRunner, Document, Callable[[dict], Awaitable[InferenceResult]]],
            Awaitable[InferenceResult | dict | list | str | float | int | bool | None],
        ],
        config: InferenceConfig,
        output_writer: DiskWriter,
        checkpoints_local_dir: str | None = None,
        records_per_chunk: int = 6000,
        metadata_key: str = "rollout_results",
    ):
        """
        Initialize the inference runner.

        Args:
            rollout_fn: Function to perform a single rollout for a document.
                Takes the InferenceRunner instance, the document, and a callback that sends a request to the server.
                Should return either an InferenceResult, a JSON-serializable value (dict, list, string, number, or bool),
                or None (if all rollouts return None, the document will be removed). The callback returns an InferenceResult and may raise
                InferenceError if the request fails.
            config: Configuration for the inference server and processing
            output_writer: Writer for saving inference results
            checkpoints_local_dir: Local directory to store checkpoints. We save individual files of records_per_chunk documents each locally as a "copy" of the output_writer documents. If a task fails, we will take the locally saved files and re-upload their documents.
            records_per_chunk: Ignored if checkpoints_local_dir is not provided. Default: 6000.
            metadata_key: Key to use for storing the rollout results in the document metadata. Default: "rollout_results".
        """
        super().__init__()

        self.config = config
        self.rollout_fn = rollout_fn
        self.metadata_key = metadata_key

        self.output_writer = output_writer

        if checkpoints_local_dir is not None:
            template = getattr(self.output_writer, "output_filename", None)
            template_str = getattr(template, "template", None)
            if not template_str or template.safe_substitute(chunk_index=0) == template_str:
                raise ValueError(
                    "Checkpoint chunking requires an output writer filename template that includes ${chunk_index}. Example: '${rank}_chunk_${chunk_index}.jsonl'"
                )

        self.checkpoint_manager = CheckpointManager(checkpoints_local_dir, records_per_chunk)

        self._server: InferenceServer | None = None
        self.metrics = MetricsKeeper(window=60 * 5)
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
            return SGLangServer(self.config)
        elif stype == "vllm":
            return VLLMServer(self.config)
        elif stype == "dummy":
            return DummyServer(self.config)
        else:
            raise ValueError(f"Unsupported server type: {stype}")

    async def _send_request(self, payload: dict, semaphore: asyncio.Semaphore) -> InferenceResult:
        """
        POST payload to the local server and return the parsed result.

        Args:
            payload: The request payload to send
            semaphore: Semaphore for controlling concurrent requests
        Returns:
            InferenceResult with response data

        Raises:
            InferenceError: If the request fails
        """
        payload["model"] = self.config.model_name_or_path
        for key, value in self.config.default_generation_params.items():
            payload.setdefault(key, value)

        # Choose endpoint based on use_chat setting
        if self.config.use_chat:
            endpoint = "/v1/chat/completions"
        else:
            endpoint = "/v1/completions"

        url = f"http://localhost:{self.server.port}{endpoint}"
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
                        self.metrics.add_metrics(failed_requests=1, requests=1)
                        self.stat_update("failed_requests", value=1, unit="request")
                        raise InferenceError(
                            payload=payload, error=f"Got BadRequestError from server: {body.decode()}"
                        )
                    elif status == 500:
                        self.queue_sizes.change_queues({"running_requests": -1})
                        self.metrics.add_metrics(failed_requests=1, requests=1)
                        self.stat_update("failed_requests", value=1, unit="request")
                        raise InferenceError(
                            payload=payload, error=f"Got InternalServerError from server: {body.decode()}"
                        )
                    elif status != 200:
                        self.queue_sizes.change_queues({"running_requests": -1})
                        self.metrics.add_metrics(failed_requests=1, requests=1)
                        self.stat_update("failed_requests", value=1, unit="request")
                        raise InferenceError(payload=payload, error=f"Error http status {status}")

                    response = json.loads(body)
                    choice = response["choices"][0]

                    # Track metrics
                    usage = response.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    self.metrics.add_metrics(
                        tokens_input=prompt_tokens,
                        tokens_output=completion_tokens,
                        requests=1,
                        successful_requests=1,
                    )
                    self.stat_update("prompt_tokens", value=prompt_tokens, unit="request")
                    self.stat_update("completion_tokens", value=completion_tokens, unit="request")
                    self.stat_update("successful_requests", value=1, unit="request")

                    # Parse response based on endpoint type
                    if self.config.use_chat:
                        text = choice["message"]["content"]
                    else:
                        text = choice["text"]

                    self.queue_sizes.change_queues({"running_requests": -1})
                    return InferenceResult(text=text, finish_reason=choice["finish_reason"], usage=usage)
                except (ConnectionError, OSError, asyncio.TimeoutError) as e:
                    # This means the server is dead likely, so we need to wait for restart
                    logger.warning(f"Client error: {type(e)} {e}")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    sleep_delay = 5 * (2**attempt)
                    await asyncio.sleep(sleep_delay)
                    attempt += 1
                except asyncio.CancelledError:
                    logger.info("Request cancelled")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    raise
                except Exception as e:
                    logger.warning(f"Unexpected error: {type(e)} {e}")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    self.metrics.add_metrics(failed_requests=1, requests=1)
                    self.stat_update("failed_requests", value=1, unit="request")
                    raise InferenceError(payload=payload, error=str(e))

            self.queue_sizes.change_queues({"running_requests": -1})
            self.metrics.add_metrics(failed_requests=1, requests=1)
            self.stat_update("failed_requests", value=1, unit="request")
            raise InferenceError(payload=payload, error=f"Failed to process request after {max_retries} attempts")

    async def _save_document(self, document: Document, output_writer_context: DiskWriter, rank: int, chunk_index: int):
        """
        Save processed document to results queue.

        Args:
            document: The processed document to save
            output_writer_context: Context manager for the output writer
            rank: Process rank identifier
            chunk_index: Chunk index to save the document to
        """
        try:
            self.stat_update("documents_processed", value=1, unit="document")

            await self.checkpoint_manager.write_document(document, rank, chunk_index, output_writer_context)

        except Exception as e:
            logger.warning(f"Failed to save document: {e}")
            self.stat_update("failed_documents", value=1, unit="document")

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
        semaphore = asyncio.Semaphore(self.config.max_concurrent_generations)
        server_task = asyncio.create_task(self.server.host_server(rank=rank))
        await self.server.wait_until_ready()
        logger.info(f"Inference server up on port {self.server.port}")

        # Start metrics reporting
        self.metrics.reset()
        metrics_task = asyncio.create_task(self.metrics_reporter(interval=self.config.metric_interval))

        async def _handle_record(
            doc: Document, rank: int, chunk_index: int, output_writer_context: DiskWriter
        ) -> None:
            """
            Process a single document through the inference pipeline.

            Args:
                doc: Document to process
                rank: Process rank identifier
                chunk_index: Chunk index for the document
                output_writer_context: Output writer context for saving documents

            Raises:
                InferenceError: If document processing fails
            """
            try:
                generate_callback = partial(self._send_request, semaphore=semaphore)
                rollout_tasks = [
                    asyncio.create_task(self.rollout_fn(self, doc, generate_callback))
                    for _ in range(self.config.rollouts_per_document)
                ]
                rollout_results = await asyncio.gather(*rollout_tasks)

                doc.metadata[self.metadata_key] = [
                    rollout_result for rollout_result in rollout_results if rollout_result is not None
                ]
                self.stat_update("successful_rollouts", value=len(doc.metadata[self.metadata_key]), unit="document")
                self.stat_update(
                    "failed_rollouts",
                    value=self.config.rollouts_per_document - len(doc.metadata[self.metadata_key]),
                    unit="document",
                )

                if len(doc.metadata[self.metadata_key]) == 0:
                    doc.metadata["__no_rollouts_remove"] = True

                await self._save_document(doc, output_writer_context, rank, chunk_index)
            except InferenceError as e:
                raise e
            except Exception as e:
                # let's propagate it
                raise InferenceError(doc, e)

        # 2. Main processing loop
        tasks_pool: set[asyncio.Task] = set()
        with self.output_writer as output_writer_context:
            # this will also upload locally cached documents to the output writer
            documents_to_skip, processed_ids = await self.checkpoint_manager.parse_existing_checkpoints(
                rank, output_writer_context
            )
            if documents_to_skip > 0:
                logger.info(
                    f"Resuming from previous checkpoint. Will skip {documents_to_skip + len(processed_ids)} already processed documents"
                )

            # process remaining documents
            record_idx = -1
            chunk_index_gen = self.checkpoint_manager.chunk_index_gen()
            async for record in self._async_data_gen(data_gen):
                record_idx += 1
                chunk_index = next(chunk_index_gen)
                # Skip documents if resuming from checkpoint
                if record_idx < documents_to_skip:
                    continue
                elif record_idx == documents_to_skip and documents_to_skip > 0:
                    logger.info(f"Skipped {documents_to_skip} documents. Resuming from chunk {chunk_index}")

                # skip already processed documents from chunks in progress
                if record.id in processed_ids:
                    processed_ids.remove(record.id)
                    continue

                # Throttle by task pool size
                while len(tasks_pool) >= self.config.max_concurrent_documents:
                    done, tasks_pool = await asyncio.wait(tasks_pool, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        await task  # Re-raises any unhandled exception

                # Add task for current record
                task = asyncio.create_task(_handle_record(record, rank, chunk_index, output_writer_context))
                tasks_pool.add(task)

            # 3. Wait for all remaining tasks to complete
            if tasks_pool:
                await asyncio.gather(*tasks_pool)
                await self.checkpoint_manager.cleanup_last_chunk(rank, chunk_index)

        # 4. shutdown inference server and metrics
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
        with self.track_time(unit="total"):
            asyncio.run(self.run_async(data, rank, world_size))
