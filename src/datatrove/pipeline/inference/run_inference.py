import argparse
import asyncio
import atexit
import base64
import datetime
import hashlib
import io
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.pipeline.media.readers.base import BaseMediaReader
from s3fs.core import S3File
import json
import logging
import multiprocessing
import os
from string import Template
from contextlib import nullcontext
import random
import re
from datatrove.io import get_datafolder, DataFolderLike
from botocore.config import Config
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cache, partial
from io import BytesIO
from urllib.parse import urlparse
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.base import Document
from datatrove.pipeline.inference.utils.data_types import PageResponse, PageResult
from typing import Literal, Optional
import boto3
import httpx
from huggingface_hub import snapshot_download
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from datatrove.pipeline.inference.utils.metrics import MetricsKeeper
from datatrove.pipeline.inference.utils.metrics import QueueSizesKeeper
from datatrove.pipeline.inference.preprocessing.query_preparator import QueryPreparator
from datatrove.pipeline.inference.utils.s3_processing import get_s3_bytes_with_backoff
from datatrove.pipeline.inference.preprocessing.readers import read_warc_bytes, read_zstd_bytes
from datatrove.pipeline.inference.servers import InferenceServer
from datatrove.pipeline.inference.servers import SGLangServer
from datatrove.pipeline.inference.servers import VLLMServer
from datatrove.pipeline.inference.servers import DummyServer
from datatrove.pipeline.inference.servers import LMDeployServer
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from loguru import logger

# Manual simple implementation of HTTP Post
# It feels strange perhaps, but httpx and aiohttp are very complex beasts
# Ex. the sessionpool in httpcore has 4 different locks in it, and I've noticed
# that at the scale of 100M+ requests, that they deadlock in different strange ways
async def apost(url, json_data):
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port or 80
    path = parsed_url.path or "/"

    writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)

        json_payload = json.dumps(json_data)
        request = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(json_payload)}\r\n"
            f"Connection: close\r\n\r\n"
            f"{json_payload}"
        )
        writer.write(request.encode())
        await writer.drain()

        # Read status line
        status_line = await reader.readline()
        if not status_line:
            raise ConnectionError("No response from server")
        status_parts = status_line.decode().strip().split(" ", 2)
        if len(status_parts) < 2:
            raise ValueError(f"Malformed status line: {status_line.decode().strip()}")
        status_code = int(status_parts[1])

        # Read headers
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode().partition(":")
            headers[key.strip().lower()] = value.strip()

        # Read response body
        if "content-length" in headers:
            body_length = int(headers["content-length"])
            response_body = await reader.readexactly(body_length)
        else:
            raise ConnectionError("Anything other than fixed content length responses are not implemented yet")

        return status_code, response_body
    except Exception as e:
        # Pass through errors
        raise e
    finally:
        # But just make sure to close the socket on your way out
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass





@dataclass
class InferenceConfig:
    server_port: int = 30024
    server_type: Literal["sglang", "vllm", "tensorrt", "lmdeploy", "dummy"] = "lmdeploy"
    model_name_or_path: str = "reducto/RolmOCR"
    model_chat_template: str = "qwen2d5-vl"
    temperature: float = 0.0
    model_max_context: int = 8192
    max_image_tokens: int = 4096
    max_concurrent_requests: int = 500
    max_concurrent_tasks: int = 50
    resize_longest_side_pixels: Optional[int] = None
    metric_interval: int = 120
    kv_quantization: Optional[int] = None
    records_per_chunk: Optional[int] = None
    prompt: Optional[str] = None

class InferenceRunner(PipelineStep):
    name = "OCR Extraction 📄"
    type = "Inference 🔍"

    def __init__(self,
                 records_readers: BaseDiskReader,
                 media_path: str,
                 config: InferenceConfig,
                 output_writer: JsonlWriter | None = None,
                 media_reader: str = "warc",
                 completions_dir: DataFolderLike | None = None,
                 ):

        super().__init__()
        self.records_readers = records_readers
        self.output_writer = output_writer
        self.config = config
        self.media_path = media_path
        self.metrics = MetricsKeeper(window=60*5)
        self.queue_sizes = QueueSizesKeeper()
        self.server: InferenceServer | None = None
        self.media_reader = media_reader
        self.completions_dir = get_datafolder(completions_dir) if completions_dir is not None else None

    async def metrics_reporter(self, interval: int = 600):
        while True:
            # Leading newlines preserve table formatting in logs
            logger.info("\n" + str(self.metrics))
            logger.info("\n" + str(self.queue_sizes))
            logger.info(str(self.stats))
            # Log every 10 minutes
            await asyncio.sleep(interval)


    def build_dolma_document(self, page_results: list[PageResult], num_pages: int, processing_error: str | None, doc_language: str, document_id: str):
        # Build the document text and page spans
        document_text = ""
        pdf_offsets = []


        # THe page_results are out of order, so we need to map them back to the original page numbers
        page_mapping = {
            page_result.page_num: page_result for page_result in page_results
        }
        for index in range(num_pages):
            page_result = page_mapping.get(index)
            if page_result is not None and page_result.response.natural_text is not None:
                # with open(f"./pages/{document_id}/{index}.png", "wb") as f:
                #     f.write(base64.b64decode(page_result.image_base64))
                content = page_result.response.natural_text
                if page_result.response.finish_reason != "stop":
                    content += f"<--- stop_reason_{page_result.response.finish_reason} --->"
            else:
                content = "<--- failed_to_process_page --->"
            content += "\n" if index < num_pages - 1 else ""
            document_text += content
            pdf_offsets.append(len(document_text))

        metadata = {
            "total_input_tokens": sum(page.input_tokens for page in page_results),
            "total_output_tokens": sum(page.output_tokens for page in page_results),
            "truncated_pages": [page.page_num for page in page_results if page.response.finish_reason == "length"],
            "failed_pages": [page.page_num for page in page_results if page.failed or page.response.finish_reason in ["repetition_sentence", "repetition_line"]],
            "page_offsets": pdf_offsets,
            "extracted_pages": len(page_results),
            "language": doc_language,
            "total_pages": num_pages,
            # "page-images": [page.image_base64 for page in page_results if page.image_base64 is not None],
        }

        if processing_error is not None:
            metadata["processing_error"] = processing_error

        return document_text, metadata



    async def ocr_page(self, page_image_b64: str, page_text: str, page_num: int, semaphore: asyncio.Semaphore) -> PageResult:
        MAX_RETRIES = 5
        attempt = 0

        query = {
            "model": "Qwen/Qwen2-VL-7B-Instruct" if self.config.server_type == "sglang" else self.config.model_name_or_path,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_image_b64}"}, "max_pixels": self.config.max_image_tokens * 28 * 28, "min_pixels": 56 * 28 * 28},
                    {"type": "text", "text": self.config.prompt if self.config.prompt else "Return the plain text representation of this document as if you were reading it naturally.\n"},
                ]},
            ],
            "max_tokens": 4096,
            "temperature": self.config.temperature,
            "repetition_penalty": 1.05,
        }

        self.queue_sizes.change_queues({"waiting_requests": 1})
        async with semaphore:
            self.queue_sizes.change_queues({"waiting_requests": -1})
            self.queue_sizes.change_queues({"running_requests": 1})
            while attempt < MAX_RETRIES:
                # Update this as the server port could have changed
                COMPLETION_URL = f"http://localhost:{self.server.port}/v1/chat/completions"
                try:
                    status_code, response_body = await apost(COMPLETION_URL, json_data=query)
                    if status_code == 400:
                        raise ValueError(f"Got BadRequestError from server: {response_body}, skipping this response")
                    elif status_code == 500:
                        raise ValueError(f"Got InternalServerError from server: {response_body}, skipping this response")
                    elif status_code != 200:
                        raise ValueError(f"Error http status {status_code}")

                    base_response_data = json.loads(response_body)

                    self.metrics.add_metrics(
                        tokens_input=base_response_data["usage"].get("prompt_tokens", 0),
                        tokens_output=base_response_data["usage"].get("completion_tokens", 0),
                    )

                    model_response_json = base_response_data["choices"][0]["message"]["content"]
                    finish_reason = base_response_data["choices"][0]["finish_reason"]
                    page_response = PageResponse(natural_text=model_response_json, finish_reason=finish_reason)

                    self.queue_sizes.change_queues({"running_requests": -1})
                    return PageResult(
                        page_num=page_num,
                        response=page_response,
                        # image_base64=page_image_b64,
                        input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
                        output_tokens=base_response_data["usage"].get("completion_tokens", 0),
                        failed=False,
                    )
                except (ConnectionError, OSError, asyncio.TimeoutError) as e:
                    logger.warning(f"Client error on attempt {attempt} for {page_num}: {type(e)} {e}")
                    # Now we want to do exponential backoff, and not count this as an actual page retry
                    # Page retrys are supposed to be for fixing bad results from the model, but actual requests to sglang
                    # are supposed to work. Probably this means that the server is just restarting
                    sleep_delay = 10 * (2**attempt)
                    logger.info(f"Sleeping for {sleep_delay} seconds on {page_num} to allow server restart")
                    await asyncio.sleep(sleep_delay)
                    attempt += 1
                except asyncio.CancelledError:
                    logger.info(f"Process page {page_num} cancelled")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    raise

            logger.error(f"Failed to process {page_num} after {MAX_RETRIES} attempts.")

            self.queue_sizes.change_queues({"running_requests": -1})
            return PageResult(
                page_num=page_num,
                response=PageResponse(
                    natural_text=None,
                    finish_reason=None,
                ),
                input_tokens=0,
                output_tokens=0,
                failed=True,
                # image_base64=page_image_b64,
            )


    async def process_pdf(self, document: Document, s3_client, query_preparator: QueryPreparator, semaphore: asyncio.Semaphore):
        # TODO implement this without opening a file every single time

        if self.media_reader == "warc":
            pdf_data, length = await asyncio.to_thread(read_warc_bytes, s3_client, f"{self.media_path}/{document.metadata['warc_filename']}", document.metadata["warc_record_offset"])
        elif self.media_reader == "zstd":
            pdf_data, length = await asyncio.to_thread(read_zstd_bytes, s3_client, f"{self.media_path}/{document.media[0].path}", document.media[0].offset)
        else:
            raise ValueError(f"Unsupported media reader: {self.media_reader}")

        page_results = []
        num_pages, doc_language, pdf_query_iterator = await query_preparator.process(pdf_data, length, image_rotation=0, id=document.id)
        
        # Process all batches and collect results
        query_tasks = set()
        processing_error = None
        try:
            async for page_num, page_image_b64, page_text in pdf_query_iterator:
                query_tasks.add(asyncio.create_task(self.ocr_page(page_image_b64, page_text, page_num, semaphore)))
        except Exception as e:
            logger.exception(f"Exception occurred while processing document {document.id}")
            processing_error = str(e)

        page_results = await asyncio.gather(*query_tasks, return_exceptions=False)
        return self.build_dolma_document(page_results, num_pages, processing_error, doc_language, document.id)

    async def process_document(self, document: Document, s3_client, query_preparator: QueryPreparator, semaphore: asyncio.Semaphore):
        try:
            document_text, metadata = await self.process_pdf(document, s3_client, query_preparator, semaphore)
            document.text = document_text
            document.media[0].metadata["pdf_metadata"] = metadata
        except Exception as e:
            logger.exception(f"Exception occurred while processing document {document.id}")
            document.media[0].metadata["pdf_metadata"]["processing_error"] = str(e)
        return document


    async def save_document(self, document: Document, rank: int, chunk_index: int | None = None):
        if self.output_writer is not None:
            await asyncio.to_thread(self.output_writer.write, document, rank=rank, chunk_index=chunk_index)
        
        failed_pages = len(document.media[0].metadata["pdf_metadata"].get("failed_pages", []))
        truncated_pages = len(document.media[0].metadata["pdf_metadata"].get("truncated_pages", []))
        success_pages = document.media[0].metadata["pdf_metadata"].get("extracted_pages", 0) - failed_pages
        all_pages = document.media[0].metadata["pdf_metadata"].get("total_pages", 0)
        document_without_errors = success_pages == all_pages and document.media[0].metadata["pdf_metadata"].get("processing_error", None) is None
        self.metrics.add_metrics(
            tokens_finished_input=document.media[0].metadata["pdf_metadata"].get("total_input_tokens", 0),
            tokens_finished_output=document.media[0].metadata["pdf_metadata"].get("total_output_tokens", 0),
            pages=success_pages + failed_pages 
        )
        self.stat_update(
            "success_pages", unit="pages", value=success_pages,
        )
        self.stat_update(
            "truncated_pages", unit="pages", value=truncated_pages,
        )
        self.stat_update(
            "failed_pages", unit="pages", value=failed_pages,
        )
        self.stat_update(
            "success_documents", value=1, unit="documents",
        )
        if not document_without_errors:
            self.stat_update(
                "partially_failed_documents", value=1, unit="documents",
            )

    async def _exhaust_task_pool(self, tasks_pool: set, rank: int, chunk_index: int | None = None):
        """Exhaust all remaining tasks in the pool and return count of processed documents"""
        documents_processed = 0
        while tasks_pool:
            results, tasks_pool = await asyncio.wait(tasks_pool, return_when=asyncio.FIRST_COMPLETED)
            for result in results:
                result_document = result.result()
                if result_document:
                    await self.save_document(result_document, rank, chunk_index)
                    documents_processed += 1
                else:
                    self.stat_update("failed_documents", value=1, unit="documents")
        return documents_processed

    def _read_checkpoint(self, rank: int) -> tuple[int, int]:
        """Read the last completed chunk index from checkpoint file"""
        if self.completions_dir is None or self.config.records_per_chunk is None:
            return -1, 0

        if self.completions_dir.exists(f"{rank}.txt"):
            return tuple(map(int, self.completions_dir.read_text(f"{rank}.txt").split("\n")))
        return -1, 0

    def _write_checkpoint(self, rank: int, chunk_index: int, total_documents_processed: int):
        """Write the completed chunk index to checkpoint file"""
        if self.completions_dir is None:
            return
        
        self.completions_dir.write_text(f"{rank}.txt", str(chunk_index) + "\n" + str(total_documents_processed))

    async def run_async(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        client_config = Config(
            max_pool_connections=self.config.max_concurrent_tasks+10,
        )
        logger.info(f"completions_dir: {self.completions_dir}, records_per_chunk: {self.config.records_per_chunk}")
        workspace_session = boto3.Session()
        workspace_s3 = workspace_session.client("s3", config=client_config)
        logger.info(f"Starting pipeline with PID {os.getpid()}")
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        if self.config.server_type == "sglang":
            self.server = SGLangServer(self.config.model_name_or_path, self.config.model_chat_template, self.config.model_max_context)
        elif self.config.server_type == "vllm":
            self.server = VLLMServer(self.config.model_name_or_path, self.config.model_chat_template, self.config.model_max_context)
        elif self.config.server_type == "lmdeploy":
            self.server = LMDeployServer(self.config.model_name_or_path, self.config.model_chat_template, self.config.model_max_context, self.config.kv_quantization)
        elif self.config.server_type == "dummy":
            self.server = DummyServer(self.config.model_name_or_path, self.config.model_chat_template, self.config.model_max_context)
        else:
            raise ValueError(f"Unsupported server type: {self.config.server_type}")

        # Start the server
        server_task = asyncio.create_task(self.server.start_server_task(semaphore, self.config.server_port, offset=rank))
        await self.server.wait_until_ready()

        logger.info(f"Server started on port {self.server.port}")
        self.metrics.reset()

        metrics_task = asyncio.create_task(self.metrics_reporter(interval=self.config.metric_interval))
        tasks_pool = set()

        # Add chunk index to the output filename
        if self.output_writer is not None and self.config.records_per_chunk is not None:
            self.output_writer.output_filename = Template(self.output_writer.output_filename.template.replace("${rank}", "${rank}_chunk${chunk_index}"))

        # Read checkpoint to determine starting point
        last_completed_chunk, total_documents_processed = self._read_checkpoint(rank)
        chunk_index = last_completed_chunk + 1
        documents_to_skip = total_documents_processed
        if documents_to_skip > 0:
            logger.info(f"Resuming from chunk {chunk_index}, will skip {documents_to_skip} already processed documents")

        total_documents_processed = 0

        with self.track_time("total_time"), (self.output_writer if self.output_writer is not None else nullcontext()):
            async with QueryPreparator(resize_longest_side_pixels=self.config.resize_longest_side_pixels, max_visual_tokens=self.config.max_image_tokens, is_zstd=self.media_reader == "zstd") as query_preparator:
                chunk_documents_read = 0
                documents_skipped = 0
                
                for record in self.records_readers.run(rank=rank, world_size=world_size):
                    # Skip documents that were already processed in previous chunks
                    if documents_skipped < documents_to_skip:
                        documents_skipped += 1
                        continue
                    chunk_documents_read += 1
                    total_documents_processed += 1
                    
                    
                    while len(tasks_pool) >= self.config.max_concurrent_tasks:
                        results, tasks_pool = await asyncio.wait(tasks_pool, return_when=asyncio.FIRST_COMPLETED, timeout=None)
                        for result in results:
                            result_document = result.result()
                            if result_document:
                                await self.save_document(result_document, rank, chunk_index)
                                
                            else:
                                self.stat_update("failed_documents", value=1, unit="documents")
                    
                    new_future = asyncio.create_task(self.process_document(record, workspace_s3, query_preparator, semaphore))
                    tasks_pool.add(new_future)
                    
                    # If records_per_chunk is set and we've processed enough, exhaust the pool
                    if self.config.records_per_chunk is not None and chunk_documents_read >= self.config.records_per_chunk:
                        # Exhaust all remaining tasks
                        await self._exhaust_task_pool(tasks_pool, rank, chunk_index)
                        tasks_pool = set()
                        
                        # Update checkpoint
                        self._write_checkpoint(rank, chunk_index, total_documents_processed)
                        
                        logger.info(f"Completed chunk {chunk_index}, processed {self.config.records_per_chunk} documents")
                        chunk_documents_read = 0  # Reset to skip count so next chunk starts correctly
                        chunk_index += 1
        
                # Process any remaining tasks
                await self._exhaust_task_pool(tasks_pool, rank, chunk_index)
                tasks_pool = set()

        self.server.cancel()
        metrics_task.cancel()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
         asyncio.run(self.run_async(data, rank, world_size))
