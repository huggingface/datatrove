import asyncio
import base64
import json
import logging
import multiprocessing as mp
import sys
from typing import AsyncGenerator, Awaitable, Tuple, Optional
from loguru import logger



class QueryWorker:
    """Manages a single worker subprocess for PDF page processing using asyncio."""
    
    def __init__(self, worker_id: int, resize_longest_side_pixels: Optional[int] = 1024, max_visual_tokens: int = 4096, timeout: float = 60.0):
        self.worker_id = worker_id
        self.timeout = timeout
        self.process: Optional[asyncio.subprocess.Process] = None
        self.resize_longest_side_pixels = resize_longest_side_pixels
        self.max_visual_tokens = max_visual_tokens
        
    async def _cleanup_process(self):
        """Clean up the worker process."""
        if self.process is not None:
            try:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    logger.warning(f"Force killing worker {self.worker_id}")
                    self.process.kill()
                    await self.process.wait()
            except:
                pass
            finally:
                self.process = None
                
    async def _ensure_process(self):
        """Ensure the worker process is running and ready."""
        if self.process is None or self.process.returncode is not None:
            if self.process is not None:
                await self._cleanup_process()
                
            # Create worker script
            # Start subprocess using asyncio
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "datatrove.pipeline.inference.preprocessing.worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                # stderr=asyncio.subprocess.,
                # 50MB
                limit=1024 * 1024 * 50
            )
            
            logger.debug(f"Worker {self.worker_id} process started with PID {self.process.pid}")
            
    async def process_document(self, zstd_data: bytes, length: int, image_rotation: int = 0, id: str = "None"):
        """Process a PDF document and return async generator for results."""
        await self._ensure_process()
        
        try:
            # Prepare input data
            input_data = {
                "zstd_data_b64": base64.b64encode(zstd_data).decode("utf-8"),
                "length": length,
                "resize_longest_side_pixels": self.resize_longest_side_pixels,
                "max_visual_tokens": self.max_visual_tokens,
                "image_rotation": image_rotation,
                "id": id
            }
            
            # Send input data to subprocess
            input_json = json.dumps(input_data)
            if self.process and self.process.stdin:
                self.process.stdin.write(f"{input_json}\n".encode())
                await self.process.stdin.drain()
            
            # Read results line by line with timeout
            async def read_with_timeout():
                if self.process and self.process.stdout:
                    while True:
                        try:
                            line = await asyncio.wait_for(
                                self.process.stdout.readline(), 
                                timeout=self.timeout
                            )
                            if not line:
                                break
                            yield line.decode().strip()
                        except asyncio.TimeoutError:
                            raise TimeoutError(f"Worker {self.worker_id} timed out")
                            
            async for line in read_with_timeout():
                if line:
                    try:
                        result = json.loads(line)
                        msg_type = result["type"]
                        data = result["data"]
                        
                        if msg_type == "complete":
                            break
                        elif msg_type == "error":
                            logger.error(f"Worker error on page {data['page_num']}: {data['error']}")
                            # Continue processing other pages
                        elif msg_type in ["num_pages", "page"]:
                            yield (msg_type, data)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse worker output: {line[:100]}")
                        
        except (TimeoutError, asyncio.TimeoutError) as e:
            logger.warning(f"Worker {self.worker_id} error: {e}")
            await self._cleanup_process()
            raise
        except BrokenPipeError as e:
            logger.warning(f"Worker {self.worker_id} broken pipe: {e}")
            await self._cleanup_process()
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in worker {self.worker_id}: {e}")
            await self._cleanup_process()
            raise
            
    async def shutdown(self):
        """Shutdown the worker process."""
        await self._cleanup_process()


class QueryPreparator:
    """Manages a pool of worker subprocesses for PDF page query preparation."""
    
    def __init__(self, num_workers: Optional[int] = None, resize_longest_side_pixels: Optional[int] = 1024, max_visual_tokens: int = 4096, timeout: float = 60.0):
        self.num_workers = num_workers or min(mp.cpu_count() // 2 + 1, 7)
        self.resize_longest_side_pixels = resize_longest_side_pixels
        self.max_visual_tokens = max_visual_tokens
        self.timeout = timeout
        self.workers = []
        self._started = False
        self._available_workers_queue = None  # Queue for available workers
        
    async def start(self):
        """Start the worker processes."""
        if self._started:
            return
            
        logger.info(f"Starting {self.num_workers} query preparator workers")
        
        
        # Create queue for available workers
        self._available_workers_queue = asyncio.Queue(maxsize=self.num_workers)
        
        for i in range(self.num_workers):
            worker = QueryWorker(i, self.resize_longest_side_pixels, self.max_visual_tokens, self.timeout)
            self.workers.append(worker)
            # Add all workers to the available queue initially
            await self._available_workers_queue.put(worker)
            
        self._started = True
        
    async def shutdown(self):
        """Shutdown all worker processes."""
        if not self._started:
            return
            
        logger.info("Shutting down query preparator workers")
        
        # Clear the queue
            # Drain the queue
        
        # Shutdown all workers concurrently
        shutdown_tasks = [worker.shutdown() for worker in self.workers]
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
        self.workers.clear()
        self._started = False
        self._available_workers_queue = None
        
    async def process(self, zstd_data: bytes, length: int, image_rotation: int = 0, id: str = "None") -> Awaitable[Tuple[int, AsyncGenerator[Tuple[int, str], None]]]:
        """
        Process a PDF and yield page queries as they become available.
        
        Args:
            zstd_data: Compressed PDF data
            length: Length of uncompressed data
            image_rotation: Rotation to apply to images (0, 90, 180, 270)
            resize_longest_side_pixels: Resize the longest side of the image to this value
            
        Yields:
            Tuple of (page_query_json, base64_png_image)
        """
        if not self._started:
            await self.start()
            
        # Get an available worker from the queue
        worker = await self._available_workers_queue.get()
        try:
            # Process document and yield results as they come
            async_iterator = worker.process_document(zstd_data, length, image_rotation,id).__aiter__()
            try:
                msg_type, data = await async_iterator.__anext__()
            except StopAsyncIteration:
                raise ValueError("No pages found")

            if not msg_type == "num_pages":
                raise ValueError(f"Expected first message to be num_pages, got {msg_type}, data: {data}")
            num_pages: int = data

            async def pages_iterator():
                try:
                    async for result in async_iterator:
                        msg_type, data = result
                        if msg_type == "page":
                            page_num = data["page_num"]
                            page_image_b64 = data["page_image_b64"]
                            page_text = data["page_text"]
                            
                            yield (page_num, page_image_b64, page_text)
                        else:
                            raise ValueError(f"Expected page message, got {msg_type}")
                except Exception as e:
                    # On error, we must reset the worker
                    logger.exception(f"Error processing PDF: {e}")
                    await worker.shutdown()
                    raise
                finally:
                    self._available_workers_queue.put_nowait(worker)
            return num_pages, pages_iterator()
                    
        except Exception as e:
            logger.exception(f"Error processing PDF: {e}")
            await worker.shutdown()
            self._available_workers_queue.put_nowait(worker)
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
        return False