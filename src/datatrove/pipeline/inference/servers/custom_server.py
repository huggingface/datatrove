import asyncio
import atexit
import logging
import sys
import os
from typing import Any

import httpx
import torch

from datatrove.pipeline.inference.servers import InferenceServer
from datatrove.utils._import_utils import check_required_dependencies
from loguru import logger

class CustomServer(InferenceServer):
    """Custom inference server implementation."""

    def __init__(self, config: "InferenceConfig"):
        """
        Initialize Custom server.

        Args:
            config: InferenceConfig containing all server configuration parameters
        """
        # Check required dependencies for Transformers server
        super().__init__(config)
        if "server_script" not in self.config.model_kwargs:
            raise ValueError("server_script is not provided in the model_kwargs")
        self.server_script = self.config.model_kwargs["server_script"]
        del self.config.model_kwargs["server_script"]

        self.server_process = None

    async def start_server_task(self) -> None:
        """Start the Transformers server process."""

        # Create the command to run the transformers_batching_app directly
        cmd = [
            sys.executable,
            self.server_script,
            "--port",
            str(self.port),
        ]

        # Add model kwargs if provided
        if self.config.model_kwargs:
            cmd.extend([f"--{k}={v}" for k, v in self.config.model_kwargs.items()])

        logger.info(f"Starting Custom server with command: {' '.join(cmd)}")

        self.server_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__))  # Run from the servers directory
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
                logger.info("Transformers server startup complete")

            # Check for common Transformers errors
            if "CUDA out of memory" in line:
                logger.error("CUDA out of memory error detected")
                if self.server_process:
                    self.server_process.terminate()
            elif "RuntimeError" in line and "CUDA" in line:
                logger.error("CUDA runtime error detected")
                if self.server_process:
                    self.server_process.terminate()
            elif "ImportError" in line or "ModuleNotFoundError" in line:
                logger.error("Import error detected - missing dependencies")
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