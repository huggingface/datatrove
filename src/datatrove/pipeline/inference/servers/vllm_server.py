import asyncio
import atexit
import logging
import sys
from typing import Any

import httpx
import torch

from datatrove.pipeline.inference.servers import InferenceServer

logger = logging.getLogger(__name__)
vllm_logger = logging.getLogger("vllm")


class VLLMServer(InferenceServer):
    """VLLM inference server implementation."""

    async def start_server_task(self, semaphore: asyncio.Semaphore, port: int, offset: int = 0) -> None:
        """Start the VLLM server process."""
        # Check GPU memory for memory settings
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        self.port = self.find_available_port(port, offset)

        cmd = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_name_or_path,
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_context),
            "--trust-remote-code",
            "--disable-log-requests",  # Disable verbose request logging
            "--disable-uvicorn-access-log",
        ]   

        
        
        # Add GPU memory fraction if needed
        if gpu_memory < 60:
            cmd.extend(["--gpu-memory-utilization", "0.80"])
        
        # Add multimodal support if needed (for vision models)
        if hasattr(self, 'chat_template') and 'vl' in self.chat_template.lower():
            cmd.extend(["--limit-mm-per-prompt", "image=1"])

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Ensure the subprocess is terminated on exit
        def _kill_proc():
            if self.process:
                self.process.terminate()

        atexit.register(_kill_proc)

        server_printed_ready_message = False

        async def process_line(line):
            nonlocal server_printed_ready_message
            vllm_logger.info(line)

            # if the server hasn't initialized yet, log all the lines to the main logger also
            if not server_printed_ready_message:
                logger.info(line)

            if "Application startup complete" in line or "Uvicorn running on" in line:
                server_printed_ready_message = True
                logger.info("VLLM server startup complete")

            # Check for common VLLM errors
            if "CUDA out of memory" in line:
                logger.error("CUDA out of memory error detected")
                if self.process:
                    self.process.terminate()
            elif "RuntimeError" in line and "CUDA" in line:
                logger.error("CUDA runtime error detected")
                if self.process:
                    self.process.terminate()

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
        stdout_task = asyncio.create_task(read_stream(self.process.stdout))
        stderr_task = asyncio.create_task(read_stream(self.process.stderr))

        try:
            await self.process.wait()
        except asyncio.CancelledError:
            logger.info("Got cancellation request for VLLM server")
            if self.process:
                self.process.terminate()
            raise

        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    async def is_ready(self) -> bool:
        """Check if the VLLM server is ready."""
        if not self.port:
            return False
        url = f"http://localhost:{self.port}/v1/models"
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url)
                return response.status_code == 200
        except Exception:
            return False 