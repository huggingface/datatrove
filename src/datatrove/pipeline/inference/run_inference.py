"""
Inference pipeline for running LLM inference on documents.

This module provides infrastructure for running inference on documents using various
inference servers like SGLang and VLLM. It supports concurrent processing, metrics
collection, and post-processing steps.

Parts of this implementation are adapted from https://github.com/allenai/olmocr
"""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from functools import partial
from typing import Awaitable, Callable, Iterable, Literal

from loguru import logger

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.checkpointing import CheckpointManager, RequestCache
from datatrove.pipeline.inference.distributed.coordination_server import CoordinationServer
from datatrove.pipeline.inference.distributed.utils import get_job_id, get_master_node_host, get_number_of_nodes
from datatrove.pipeline.inference.metrics import MetricsKeeper, QueueSizesKeeper
from datatrove.pipeline.inference.servers import (
    DummyServer,
    InferenceServer,
    SGLangServer,
    VLLMServer,
)
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
        coordination_port: Port of the coordination server used for distributed settings. (default: 9811)
        distributed_init_timeout: Timeout in seconds for distributed initialization (default: 300).
            Applies to both master (waiting for workers) and workers (connecting to cluster).
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
    # distributed
    master_port: int = 9810
    distributed_init_timeout: int = 300
    coordination_port: int = 9811
    auto_restart_server: bool = True

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
        else:
            raise ValueError(f"Unsupported server type: {stype}")

    async def _send_request(
        self, server: InferenceServer, payload: dict, semaphore: asyncio.Semaphore
    ) -> InferenceResult:
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

        max_retries = 6
        attempt = 0

        self.queue_sizes.change_queues({"waiting_requests": 1})
        async with semaphore:
            self.queue_sizes.change_queues({"waiting_requests": -1})
            self.queue_sizes.change_queues({"running_requests": 1})

            while attempt < max_retries:
                try:
                    status, body = await server.send_request(endpoint, payload)
                    if status == 400:
                        self.queue_sizes.change_queues({"running_requests": -1})
                        self.metrics.add_metrics(failed_requests=1, requests=1)
                        self.stat_update("failed_requests", value=1, unit="request")
                        raise InferenceError(
                            None, payload=payload, error=f"Got BadRequestError from server: {body.decode()}"
                        )
                    elif status == 500:
                        self.queue_sizes.change_queues({"running_requests": -1})
                        self.metrics.add_metrics(failed_requests=1, requests=1)
                        self.stat_update("failed_requests", value=1, unit="request")
                        raise InferenceError(
                            None, payload=payload, error=f"Got InternalServerError from server: {body.decode()}"
                        )
                    elif status != 200:
                        self.queue_sizes.change_queues({"running_requests": -1})
                        self.metrics.add_metrics(failed_requests=1, requests=1)
                        self.stat_update("failed_requests", value=1, unit="request")
                        raise InferenceError(None, payload=payload, error=f"Error http status {status}")

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
                    raise InferenceError(None, payload=payload, error=str(e))

            self.queue_sizes.change_queues({"running_requests": -1})
            self.metrics.add_metrics(failed_requests=1, requests=1)
            self.stat_update("failed_requests", value=1, unit="request")
            raise InferenceError(payload=payload, error=f"Failed to process request after {max_retries} attempts")

    async def _cached_request(
        self,
        payload: dict,
        semaphore: asyncio.Semaphore,
        server: InferenceServer,
        doc_id: str,
        rollout_idx: int,
        chunk_index: int,
    ) -> InferenceResult:
        """
        Return cached inference result when available, otherwise fetch from server and persist it.
        """
        if not self.request_cache.enabled:
            return await self._send_request(server, payload, semaphore)

        payload_hash = self.request_cache.prepare_payload(payload)
        cached_result, cached_error = await self.request_cache.get_cached_response(
            doc_id, rollout_idx, payload_hash=payload_hash
        )
        if cached_result is not None or cached_error is not None:
            if cached_error is not None:
                raise InferenceError(None, cached_error, payload=payload)
            return InferenceResult(
                text=cached_result["text"], finish_reason=cached_result["finish_reason"], usage=cached_result["usage"]
            )

        try:
            result = await self._send_request(server, payload, semaphore)
        except InferenceError as e:
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

        master_node_host = get_master_node_host()

        coordination_server_ctx = (
            CoordinationServer(master_node_host, self.config.coordination_port, get_job_id())
            if get_number_of_nodes() > 1
            else nullcontext()
        )

        async with (
            coordination_server_ctx as coordination_server,
            self._init_server(rank=rank) as (inference_server, is_master_node),
        ):
            # In the distributed regime, the participating nodes are excpected to do some sort of weak barrier (ray start/join for vllm, torch.distributed.barrier for sglang) during the init_server..
            # We can thus expect that by this point, the coordination server must be running on master node for all worker nodes.
            # This is not true for the master node, which might not have started the coordination server yet.
            if not is_master_node:
                while coordination_server is not None and await coordination_server.master_running():
                    await asyncio.sleep(10)
                else:
                    logger.info("Master node is not running, exiting")
            else:
                # Start metrics reporting
                self.metrics.reset()
                metrics_task = asyncio.create_task(self.metrics_reporter(interval=self.config.metric_interval))

                await self.request_cache.initialize(rank)

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
                        rollout_tasks = []
                        for rollout_idx in range(self.config.rollouts_per_document):
                            generate_callback = partial(
                                self._cached_request,
                                semaphore=semaphore,
                                server=inference_server,
                                doc_id=str(doc.id),
                                rollout_idx=rollout_idx,
                                chunk_index=chunk_index,
                            )
                            rollout_tasks.append(asyncio.create_task(self.rollout_fn(self, doc, generate_callback)))
                        rollout_results = await asyncio.gather(*rollout_tasks)

                        doc.metadata[self.metadata_key] = [
                            rollout_result for rollout_result in rollout_results if rollout_result is not None
                        ]
                        self.stat_update(
                            "successful_rollouts", value=len(doc.metadata[self.metadata_key]), unit="document"
                        )
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
                    except Exception as e:
                        # let's propagate it
                        raise InferenceError(doc, e)

                # 2. Main processing loop
                tasks_pool: set[asyncio.Task] = set()
                chunk_index: int | None = None
                completed_successfully = False
                try:
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
                                logger.info(
                                    f"Skipped {documents_to_skip} documents. Resuming from chunk {chunk_index}"
                                )

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
                                _handle_record(record, rank, chunk_index, output_writer_context)
                            )
                            tasks_pool.add(task)

                        # 3. Wait for all remaining tasks to complete
                        if tasks_pool:
                            await asyncio.gather(*tasks_pool)
                            if chunk_index is not None:
                                await self.checkpoint_manager.cleanup_last_chunk(rank, chunk_index)
                    completed_successfully = True
                finally:
                    # 4. shutdown inference server and metrics
                    try:
                        metrics_task.cancel()
                        await asyncio.wait_for(metrics_task, timeout=10.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        logger.warning("Metrics task did not complete within timeout, forcing shutdown")
                    with suppress(Exception):
                        await self.request_cache.close(delete_file=completed_successfully)

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
