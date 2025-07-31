import asyncio
import atexit
import logging
import sys
from typing import Any

import httpx
import torch

from datatrove.pipeline.inference.servers import InferenceServer
from datatrove.utils._import_utils import check_required_dependencies
from loguru import logger

class VLLMServer(InferenceServer):
    """VLLM inference server implementation."""

    def __init__(self, model_name_or_path: str, max_context: int, model_kwargs: dict | None = None, server_log_folder: str | None = None):
        """
        Initialize VLLM server.
        
        Args:
            model_name_or_path: Path or name of the model to load
            max_context: Maximum context length for the model
            model_kwargs: Additional keyword arguments for model initialization
            server_log_folder: Optional directory path where server logs will be stored
        """
        # Check required dependencies for VLLM server
        check_required_dependencies("VLLM server", ["vllm"])
        super().__init__(model_name_or_path, max_context, model_kwargs, server_log_folder)

    async def start_server_task(self) -> None:
        """Start the VLLM server process."""

        cmd = [
            "vllm",
            "serve",
            self.model_name_or_path,
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_context),
            "--trust-remote-code",
            "--disable-log-requests",  # Disable verbose request logging
            "--disable-uvicorn-access-log",
        ]   

        if self.model_kwargs:
            cmd.extend([f"--{k}={v}" for k, v in self.model_kwargs.items()])

        self.server_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Ensure the subprocess is terminated on exit
        def _kill_proc():
            if self.server_process:
                self.server_process.terminate()

        atexit.register(_kill_proc)

        server_printed_ready_message = False
        
        # Create dedicated logger for server output
        server_logger = self._create_server_logger(getattr(self, 'rank', 0))

        async def process_line(line):
            nonlocal server_printed_ready_message

            # Always log to file if server logger is available
            if server_logger:
                server_logger.info(line)

            # if the server hasn't initialized yet, log all the lines to the main logger also
            if not server_printed_ready_message:
                logger.info(line)

            if "Application startup complete" in line or "Uvicorn running on" in line:
                server_printed_ready_message = True
                logger.info("VLLM server startup complete")

            # Check for common VLLM errors
            if "CUDA out of memory" in line:
                logger.error("CUDA out of memory error detected")
                if self.server_process:
                    self.server_process.terminate()
            elif "RuntimeError" in line and "CUDA" in line:
                logger.error("CUDA runtime error detected")
                if self.server_process:
                    self.server_process.terminate()

        async def read_stream(stream):
            while True:
                line = await stream.readline()
                if not line:
                    break
                try:
                    line = line.decode("utf-8").rstrip()
                    await process_line(line)
                except Exception as ex:
                    logger.warning(f"Got {ex} when reading log line from inference server, skipping")

        # Start tasks to read stdout and stderr
        stdout_task = asyncio.create_task(read_stream(self.server_process.stdout))
        stderr_task = asyncio.create_task(read_stream(self.server_process.stderr))

        try:
            await self.server_process.wait()
        except asyncio.CancelledError:
            if self.server_process:
                self.server_process.terminate()
            raise

        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
