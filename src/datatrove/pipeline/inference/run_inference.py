"""
Inference pipeline for running LLM inference on documents.

This module provides infrastructure for running inference on documents using various
inference servers like SGLang and VLLM. It supports concurrent processing, metrics
collection, and post-processing steps.

Parts of this implementation are adapted from https://github.com/allenai/olmocr
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import partial
from typing import AsyncIterator, Callable, ContextManager, Iterable, Literal

from loguru import logger

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.checkpointing import CheckpointManager, RequestCache
from datatrove.pipeline.inference.metrics import MetricsKeeper, QueueSizesKeeper
from datatrove.pipeline.inference.servers import (
    CustomServer,
    DummyServer,
    EndpointServer,
    InferenceServer,
    SGLangServer,
    VLLMServer,
)
from datatrove.pipeline.inference.types import InferenceError, InferenceResult, RolloutFunction, ServerError
from datatrove.pipeline.writers.disk_base import DiskWriter


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
        master_port: Port of the master node used for distributed settings. (default: 9810)
    """

    # server and model
    server_type: Literal["sglang", "vllm", "dummy", "custom", "endpoint"]
    model_name_or_path: str
    model_max_context: int = 8192
    use_chat: bool = True
    endpoint_url: str | None = None  # Required when server_type is "endpoint"
    api_key: str | None = None  # API key for endpoint authentication (Bearer token)
    # metrics
    metric_interval: int = 120
    # parallelism
    tp: int = 1
    dp: int = 1
    pp: int = 1
    # rollouts and generation
    default_generation_params: dict = field(default_factory=dict)
    rollouts_per_document: int = 1
    max_concurrent_generations: int = 500
    max_concurrent_documents: int | None = None
    request_timeout: float | None = None  # Timeout for HTTP requests in seconds (None means no timeout)
    # other
    model_kwargs: dict | None = None
    server_log_folder: str | None = None
    # distributed, we could probably init inside the server class, so that we can run multiple jobs on same nodes, but this use-case
    # doesn't make much sense, so keep it as is now.
    master_port: int = 9810

    def __post_init__(self):
        if self.max_concurrent_documents is None:
            self.max_concurrent_documents = max(1, self.max_concurrent_generations // self.rollouts_per_document)


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
        rollout_fn: RolloutFunction,
        config: InferenceConfig,
        output_writer: DiskWriter,
        shared_context: (dict | Callable[[], dict] | ContextManager[dict] | None) = None,
        checkpoints_local_dir: str | None = None,
        records_per_chunk: int = 6000,
        metadata_key: str = "rollout_results",
    ):
        """
        Initialize the inference runner.

        Args:
            rollout_fn: Function to perform a single rollout for a document.
                Takes the document and a callback that sends a request to the server.
                Additionally receives keyword arguments from shared_context.
                Should return either an InferenceResult, a JSON-serializable value (dict, list, string, number, or bool),
                or None (if all rollouts return None, the document will be removed). The callback returns an InferenceResult and may raise
                InferenceError if the request fails.
            config: Configuration for the inference server and processing
            output_writer: Writer for saving inference results
            shared_context: Shared context to be used by the rollout function. Can be a dict, a callable that returns a dict, or a context manager that returns a dict. The dict items will be passed as keyword arguments to the rollout_fn.
            checkpoints_local_dir: Local directory to store checkpoints. We save individual files of records_per_chunk documents each locally as a "copy" of the output_writer documents. If a task fails, we will take the locally saved files and re-upload their documents.
            records_per_chunk: Ignored if checkpoints_local_dir is not provided. Default: 6000.
            metadata_key: Key to use for storing the rollout results in the document metadata. Default: "rollout_results".
        """
        super().__init__()

        self.config = config
        self.rollout_fn = rollout_fn
        self.shared_context = shared_context
        self.metadata_key = metadata_key

        self.output_writer = output_writer

        if checkpoints_local_dir is not None:
            template = getattr(self.output_writer, "output_filename", None)
            template_str = getattr(template, "template", None)
            if not template_str or template.safe_substitute(chunk_index=0) == template_str:
                raise ValueError(
                    "Checkpoint chunking requires an output writer filename template that includes ${chunk_index}. Example: '${rank}_chunk_${chunk_index}.jsonl'"
                )

        self.request_cache = RequestCache(checkpoints_local_dir)
        self.checkpoint_manager = CheckpointManager(
            checkpoints_local_dir, records_per_chunk, request_cache=self.request_cache
        )

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

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _init_server(self, rank: int) -> InferenceServer:
        """
        Spawn the requested inference server (non-blocking).

        Returns:
            The initialized inference server instance

        Raises:
            ValueError: If unsupported server type is specified
        """
        stype = self.config.server_type

        if stype == "sglang":
            return SGLangServer(self.config, rank)
        elif stype == "vllm":
            return VLLMServer(self.config, rank)
        elif stype == "dummy":
            return DummyServer(self.config, rank)
        elif stype == "custom":
            return CustomServer(self.config, rank)
        elif stype == "endpoint":
            return EndpointServer(self.config, rank)
        else:
            raise ValueError(f"Unsupported server type: {stype}")

    async def _send_request(
        self, server: InferenceServer, payload: dict, semaphore: asyncio.Semaphore
    ) -> InferenceResult:
        """
        POST payload to the server and return the parsed result.

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

        max_retries = 6
        attempt = 0

        self.queue_sizes.change_queues({"waiting_requests": 1})
        async with semaphore:
            self.queue_sizes.change_queues({"waiting_requests": -1})
            self.queue_sizes.change_queues({"running_requests": 1})

            while attempt < max_retries:
                try:
                    response = await server.make_request(payload)
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
                    # This means the server is dead likely, let's try again just to be sure
                    logger.warning(f"Client error: {type(e)} {e}")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    sleep_delay = 5 * (2**attempt)
                    await asyncio.sleep(sleep_delay)
                    attempt += 1
                except (asyncio.CancelledError, ServerError):
                    logger.info("Request cancelled")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    raise
                except InferenceError:
                    # Re-raise InferenceError as-is
                    self.queue_sizes.change_queues({"running_requests": -1})
                    self.metrics.add_metrics(failed_requests=1, requests=1)
                    self.stat_update("failed_requests", value=1, unit="request")
                    raise
                except Exception as e:
                    logger.warning(f"Unexpected error: {type(e)} {e}")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    self.metrics.add_metrics(failed_requests=1, requests=1)
                    self.stat_update("failed_requests", value=1, unit="request")
                    raise InferenceError(None, str(e), payload=payload)

            self.queue_sizes.change_queues({"running_requests": -1})
            self.metrics.add_metrics(failed_requests=1, requests=1)
            self.stat_update("failed_requests", value=1, unit="request")
            raise InferenceError(None, f"Failed to process request after {max_retries} attempts", payload=payload)

    async def _cached_request(
        self,
        payload: dict,
        semaphore: asyncio.Semaphore,
        server: InferenceServer,
        doc_id: str,
        rollout_idx: int,
        chunk_index: int,
        request_counters: list[int] | None = None,
    ) -> InferenceResult:
        """
        Return cached inference result when available, otherwise fetch from server and persist it.
        """
        if request_counters is not None:
            request_counters[rollout_idx] += 1
        if not self.request_cache.enabled:
            return await self._send_request(server, payload, semaphore)

        payload_hash = self.request_cache.prepare_payload(payload)
        cached_result, cached_error = await self.request_cache.get_cached_response(
            doc_id, rollout_idx, payload_hash=payload_hash
        )
        if cached_result is not None or cached_error is not None:
            if cached_error is not None and "BadRequestError" in cached_error:
                raise InferenceError(None, cached_error, payload=payload)
            elif cached_result is not None:
                return InferenceResult(
                    text=cached_result["text"],
                    finish_reason=cached_result["finish_reason"],
                    usage=cached_result["usage"],
                )

        try:
            result = await self._send_request(server, payload, semaphore)
        except InferenceError as e:
            if "BadRequestError" in str(e):
                await self.request_cache.store_error(
                    chunk_index=chunk_index,
                    doc_id=doc_id,
                    rollout_idx=rollout_idx,
                    error_message=str(e),
                    payload_hash=payload_hash,
                )
            raise

        await self.request_cache.store_result(
            chunk_index=chunk_index,
            doc_id=doc_id,
            rollout_idx=rollout_idx,
            result={"text": result.text, "finish_reason": result.finish_reason, "usage": result.usage},
            payload_hash=payload_hash,
        )
        return result

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

    async def _async_data_gen(self, sync_gen: Iterable[Document]) -> AsyncIterator[Document]:
        """
        Convert synchronous generator to async generator using a threadpool.

        Args:
            sync_gen: Synchronous iterable of documents

        Yields:
            Document objects from the synchronous generator
        """

        # One thread, so that we don't instantiate a new thread for each document
        threadpool = ThreadPoolExecutor(max_workers=1)
        try:

            def get_next_item(iterator: Iterable[Document]):
                try:
                    return next(iterator)
                except StopIteration:
                    return

            iterator = iter(sync_gen)
            loop = asyncio.get_running_loop()
            while True:
                item = await loop.run_in_executor(threadpool, get_next_item, iterator)
                if item is None:
                    break
                yield item
        finally:
            await asyncio.wait_for(asyncio.to_thread(threadpool.shutdown, wait=False, cancel_futures=True), timeout=10)

    @contextmanager
    def get_shared_context_cm(self) -> dict:
        if self.shared_context is None:
            yield {}
        elif isinstance(self.shared_context, dict):
            yield self.shared_context
        elif hasattr(self.shared_context, "__enter__") and hasattr(self.shared_context, "__exit__"):
            # Check for context manager before callable, since @contextmanager objects are also callable
            with self.shared_context as shared_context:
                yield shared_context or {}
        elif callable(self.shared_context):
            # Check if it's a context manager factory by seeing if calling it returns one
            result = self.shared_context()
            if hasattr(result, "__enter__") and hasattr(result, "__exit__"):
                # It's a context manager factory (like @contextmanager decorated function)
                with result as shared_context:
                    yield shared_context or {}
            else:
                # It's a regular callable that returns a dict
                yield result or {}
        else:
            raise ValueError(f"Invalid shared context type: {type(self.shared_context)}")

    async def _handle_record(
        self,
        doc: Document,
        rank: int,
        chunk_index: int,
        rollout_fn: RolloutFunction,
        output_writer_context: DiskWriter,
        server: InferenceServer,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """
        Process a single document through the inference pipeline.

        Args:
            doc: Document to process
            rank: Process rank identifier
            chunk_index: Chunk index for the document
            rollout_fn: Function to perform a single rollout for a document. Already wrapped with the shared context.
            output_writer_context: Output writer context for saving documents
            server: Inference server to use
            semaphore: Semaphore for controlling concurrent requests
        """
        try:
            request_counters = [0] * self.config.rollouts_per_document
            rollout_tasks = []
            for rollout_idx in range(self.config.rollouts_per_document):
                generate_callback = partial(
                    self._cached_request,
                    semaphore=semaphore,
                    server=server,
                    doc_id=str(doc.id),
                    rollout_idx=rollout_idx,
                    chunk_index=chunk_index,
                    request_counters=request_counters,
                )
                rollout_tasks.append(asyncio.create_task(rollout_fn(doc, generate_callback)))
            rollout_results = await asyncio.gather(*rollout_tasks)

            for count in request_counters:
                self.stat_update("requests", value=count, unit="rollout")

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
            await self.request_cache.mark_document_complete(doc.id)
        except InferenceError as e:
            raise e
        except (ServerError, asyncio.CancelledError):
            raise
        except Exception as e:
            # let's propagate it
            raise InferenceError(doc, e)

    # --------------------------------------------------------------------- #
    # Async processing
    # --------------------------------------------------------------------- #
    async def run_async(
        self,
        data_gen: Iterable[Document],
        rank: int = 0,
    ) -> None:
        """
        Run asynchronous inference processing on the provided data.

        Args:
            data_gen: Iterable of Document objects to process
            rank: Process rank identifier for distributed processing
        """
        async with self._init_server(rank=rank) as inference_server:
            if not inference_server:
                # we are a worker, we do not send requests
                return

            # 1. Initialize semaphore and request cache
            semaphore = asyncio.Semaphore(self.config.max_concurrent_generations)

            await self.request_cache.initialize(rank)

            # Start metrics reporting
            self.metrics.reset()
            metrics_task = asyncio.create_task(self.metrics_reporter(interval=self.config.metric_interval))

            # 2. Main processing loop
            tasks_pool: set[asyncio.Task] = set()
            chunk_index: int | None = None
            completed_successfully = False
            try:
                with self.output_writer as output_writer_context, self.get_shared_context_cm() as shared_context_data:
                    # Wrap the rollout function with the shared context
                    rollout_fn: RolloutFunction = partial(self.rollout_fn, **shared_context_data)

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
                        task = asyncio.create_task(
                            self._handle_record(
                                record,
                                rank,
                                chunk_index,
                                rollout_fn,
                                output_writer_context,
                                inference_server,
                                semaphore,
                            )
                        )
                        tasks_pool.add(task)

                    # 3. Wait for all remaining tasks to complete
                    if tasks_pool:
                        await asyncio.gather(*tasks_pool)
                        if chunk_index is not None:
                            await self.checkpoint_manager.cleanup_last_chunk(rank, chunk_index)
                completed_successfully = True
            finally:
                await self._cleanup(metrics_task, tasks_pool, delete_cache_file=completed_successfully)

    async def _cleanup(
        self,
        metrics_task: asyncio.Task,
        tasks_pool: set[asyncio.Task],
        delete_cache_file: bool,
        timeout: int = 30,
    ) -> None:
        async def _wait_for_tasks_to_complete():
            with suppress(Exception):
                await asyncio.wait_for(self.request_cache.close(delete_file=delete_cache_file), timeout=timeout // 2)
            await asyncio.gather(metrics_task, *tasks_pool, return_exceptions=True)

        # First cancel all tasks in the pool
        for task in tasks_pool:
            task.cancel()

        # Cancel the metrics task
        metrics_task.cancel()
        try:
            await asyncio.wait_for(_wait_for_tasks_to_complete(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timed out while cleaning up inference runner.")

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
            asyncio.run(self.run_async(data, rank))
