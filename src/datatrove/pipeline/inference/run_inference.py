"""
Minimal inference pipeline.

The runner:
  • pulls records from one or more PipelineStep readers
  • converts each record into an LLM request via `query_builder`
  • sends the request to a locally spawned inference server
  • returns / logs the raw model response
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from string import Template
from typing import Callable, Optional, Sequence, Union, Iterable, Literal

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.servers import (
    DummyServer,
    InferenceServer,
    LMDeployServer,
    SGLangServer,
    VLLMServer,
)
from datatrove.data import Document, DocumentsPipeline
from loguru import logger

# --------------------------------------------------------------------------- #
# Low-level, dependency-free HTTP POST helper (kept from the original file)
# --------------------------------------------------------------------------- #
async def _raw_post(url: str, json_data: dict) -> tuple[int, bytes]:
    """Very small HTTP/1.1 POST helper using the std-lib socket machinery."""
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
    server_port: int = 30024
    server_type: Literal["sglang", "vllm", "tensorrt", "lmdeploy", "dummy"] = "lmdeploy"
    model_name_or_path: str = "reducto/RolmOCR"
    model_chat_template: str = "qwen2d5-vl"
    temperature: float = 0.0
    model_max_context: int = 8192
    max_concurrent_requests: int = 500
    max_concurrent_tasks: int = 50
    metric_interval: int = 120
    records_per_chunk: Optional[int] = None
    model_kwargs: Optional[dict] = None
    kv_quantization: bool = False  # Add missing attribute for LMDeployServer

    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}


# --------------------------------------------------------------------------- #
# Minimal inference runner
# --------------------------------------------------------------------------- #
class InferenceRunner(PipelineStep):
    name = "Inference 🔍"
    type = "Model call"

    def __init__(
        self,
        query_builder: Callable[[Document], dict],
        config: InferenceConfig,
        post_process_steps: Union[PipelineStep, Sequence[PipelineStep]],
        completions_dir: DataFolderLike | None = None,
    ):
        super().__init__()

        # ---------------------------------------------------------------- #
        # Post-processors storage
        # ---------------------------------------------------------------- #
        # normalise to a list
        if isinstance(post_process_steps, Sequence) and not isinstance(post_process_steps, (str, bytes)):
            self.post_process_steps = list(post_process_steps)
        else:
            self.post_process_steps = [post_process_steps]
        # ---------------------------------------------------------------- #

        self.query_builder = query_builder
        self.config = config
        self.completions_dir = (
            get_datafolder(completions_dir) if completions_dir else None
        )
        self._server: InferenceServer | None = None

    @property
    def server(self) -> InferenceServer:
        """Lazy initialization of the inference server."""
        if self._server is None:
            self._server = self._init_server()
        # At this point _server is guaranteed to be not None after _init_server()
        assert self._server is not None
        return self._server

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _init_server(self) -> InferenceServer:
        """Spawn the requested inference server (non-blocking)."""
        stype = self.config.server_type
        if stype == "sglang":
            return SGLangServer(
                self.config.model_name_or_path,
                self.config.model_chat_template,
                self.config.model_max_context,
                self.config.model_kwargs,
            )
        elif stype == "vllm":
            return VLLMServer(
                self.config.model_name_or_path,
                self.config.model_chat_template,
                self.config.model_max_context,
                self.config.model_kwargs,
            )
        elif stype == "lmdeploy":
            return LMDeployServer(
                self.config.model_name_or_path,
                self.config.model_chat_template,
                self.config.model_max_context,
                self.config.model_kwargs,
            )
        elif stype == "dummy":
            # DummyServer has a different constructor signature
            return DummyServer(
                self.config.model_name_or_path,
                self.config.server_port,
                self.config.model_kwargs,
            )
        else:
            raise ValueError(f"Unsupported server type: {stype}")

    async def _send_request(self, payload: dict, semaphore: asyncio.Semaphore) -> dict | None:
        """POST payload to the local server and return the parsed JSON response."""
        url = f"http://localhost:{self.server.port}/v1/chat/completions"
        async with semaphore:
            status, body = await _raw_post(url, json_data=payload)
            if status != 200:
                logger.warning("Server returned status %s – skipping", status)
                return None
            return json.loads(body)

    def _read_checkpoint(self, rank: int) -> tuple[int, int]:
        """Read the last completed chunk index from checkpoint file"""
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
        """Write the completed chunk index to checkpoint file"""
        if self.completions_dir is None:
            return
        
        checkpoint_file = f"{rank}.txt"
        content = f"{chunk_index}\n{total_documents_processed}"
        self.completions_dir.write_text(checkpoint_file, content)

    async def _exhaust_task_pool(self, tasks_pool: set, rank: int, chunk_index: int | None = None):
        """Exhaust all remaining tasks in the pool and return count of processed documents"""
        documents_processed = 0
        while tasks_pool:
            done, tasks_pool = await asyncio.wait(tasks_pool, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    result_document = task.result()
                    if result_document:
                        await self._save_document(result_document, rank, chunk_index if self.config.records_per_chunk else None)
                        documents_processed += 1
                    else:
                        self.stat_update("failed_documents", value=1, unit="documents")
                except Exception as e:
                    logger.exception(f"Error processing document: {e}")
                    self.stat_update("failed_documents", value=1, unit="documents")
        return documents_processed

    async def _save_document(self, document: Document, rank: int, chunk_index: int | None = None):
        """Save processed document through post-processing pipeline"""
        # Add chunk_index to document metadata if chunking is enabled
        if chunk_index is not None:
            document.metadata["chunk_index"] = chunk_index

        # Run through post-processing pipeline
        tmp_gen = (d for d in [document])
        for step in self.post_process_steps:
            tmp_gen = step.run(tmp_gen, rank, world_size=1)

        # Exhaust the generator to trigger all post-processing steps
        for _ in tmp_gen:
            pass

    async def _async_data_gen(self, sync_gen: Iterable[Document]):
        """Convert synchronous generator to async generator using asyncio.to_thread"""
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
        logger.info(f"completions_dir: {self.completions_dir}, records_per_chunk: {self.config.records_per_chunk}")
        
        # 1. start server
        self._init_server()
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        server_task = asyncio.create_task(
            self.server.start_server_task(semaphore, offset=rank)
        )
        await self.server.wait_until_ready()
        logger.info("Inference server up on port %s", self.server.port)

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
        total_docs_processed = 0

        async def _handle_record(doc: Document):
            payload = self.query_builder(doc)
            payload.setdefault("model", self.config.model_name_or_path)
            payload.setdefault("temperature", self.config.temperature)

            rsp = await self._send_request(payload, semaphore)
            if rsp is None:
                return None
            
            # write the model answer back into the document
            choice = rsp["choices"][0]
            doc.text = choice["message"]["content"]
            doc.metadata["finish_reason"] = choice["finish_reason"]
            doc.metadata["usage"] = rsp.get("usage", {})

            return doc

        # 4. Main processing loop - unified for both chunked and non-chunked, now async
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
                        if result_document:
                            await self._save_document(result_document, rank, chunk_index if self.config.records_per_chunk else None)
                        else:
                            self.stat_update("failed_documents", value=1, unit="documents")
                    except Exception as e:
                        logger.exception(f"Error processing document: {e}")
                        self.stat_update("failed_documents", value=1, unit="documents")

            # Add task for current record
            tasks_pool.add(asyncio.create_task(_handle_record(record)))
            
            # Update counters
            if self.config.records_per_chunk is not None:
                chunk_documents_read += 1
                total_docs_processed += 1
                
                # Check if chunk is complete
                if chunk_documents_read >= self.config.records_per_chunk:
                    # Exhaust all remaining tasks for this chunk
                    await self._exhaust_task_pool(tasks_pool, rank, chunk_index)
                    tasks_pool = set()
                    
                    # Update checkpoint and prepare for next chunk
                    self._write_checkpoint(rank, chunk_index, total_docs_processed + documents_to_skip)
                    logger.info(f"Completed chunk {chunk_index}, processed {self.config.records_per_chunk} documents")
                    
                    chunk_documents_read = 0
                    chunk_index += 1

        # 5. Process any remaining tasks
        await self._exhaust_task_pool(tasks_pool, rank, chunk_index)

        # 6. shutdown inference server
        if self._server:
            self._server.cancel()
        server_task.cancel()

        return None  # explicit

    # --------------------------------------------------------------------- #
    # Synchronous entrypoint required by PipelineStep
    # --------------------------------------------------------------------- #
    def run(
        self,
        data: Iterable[Document],
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Consume `data`, run inference and post-processing, do not yield further documents."""
        asyncio.run(self.run_async(data, rank, world_size))
        return None
