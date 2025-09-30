import asyncio
import atexit
import logging
import sys
from typing import Any

import httpx
import os
import torch
from loguru import logger

from datatrove.pipeline.inference.servers import InferenceServer

# TODO: Should restart the server if it crashes

class LMDeployServer(InferenceServer):
    """LMDeploy inference server implementation."""
    async def start_server_task(self) -> None:
        """Start the LMDeploy server process."""
        # Check GPU memory for memory settings
        # Note: self.port is already set by base class host_server() method
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        
        cmd = [
            "lmdeploy",
            "serve",
            "api_server",
            self.model_name_or_path,
            "--server-port",
            str(self.port),
            "--session-len",
            str(self.max_context),
            "--chat-template",
            self.chat_template,
            "--log-level",
            "ERROR",
            "--disable-fastapi-docs",
            "--cache-max-entry-count",
            "0.9",
            "--vision-max-batch-size",
            "128",
            # "--max-prefill-token-num",
            # "32768",
            # "--cache-max-entry-count",
            # "0.8",  # Use 80% of GPU memory for KV cache
            # "--trust-remote-code",
        ]
        
        # Add tensor parallelism if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            cmd.extend(["--tp", str(torch.cuda.device_count())])
        
        # Adjust cache for smaller GPUs
        if gpu_memory < 60:
            cmd.extend(["--cache-max-entry-count", "0.6"])

        os.environ["UVICORN_LOG_LEVEL"] = "error"
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
        def process_line(line):
            nonlocal server_printed_ready_message
            # if the server hasn't initialized yet, log all the lines to the main logger also
            if not server_printed_ready_message:
                logger.info(line)

            if "Application startup complete" in line or "Uvicorn running on" in line or "server started" in line:
                server_printed_ready_message = True
                logger.info("LMDeploy server startup complete")

            # Check for common LMDeploy errors
            if "CUDA out of memory" in line:
                logger.error("CUDA out of memory error detected")
                if self.server_process:
                    self.server_process.terminate()
            elif "RuntimeError" in line and "CUDA" in line:
                logger.error("CUDA runtime error detected")
                if self.server_process:
                    self.server_process.terminate()
            elif "Failed to load model" in line:
                logger.error("Model loading failed")
                if self.server_process:
                    self.server_process.terminate()

        async def read_stream(stream):
            while True:
                line = await stream.readline()
                if not line:
                    break
                try:
                    line = line.decode("utf-8").rstrip()
                    process_line(line)
                except Exception as ex:
                    logger.warning(f"Got {ex} when reading log line from inference server, skipping")

        # Start tasks to read stdout and stderr
        stdout_task = asyncio.create_task(read_stream(self.server_process.stdout))
        stderr_task = asyncio.create_task(read_stream(self.server_process.stderr))

        try:
            await self.server_process.wait()
        except asyncio.CancelledError:
            logger.info("Got cancellation request for LMDeploy server")
            if self.server_process:
                self.server_process.terminate()
            raise

        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    async def is_ready(self) -> bool:
        """Check if the LMDeploy server is ready."""
        if not self.port:
            return False

        url = f"http://localhost:{self.port}/v1/models"
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url, timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False 