import asyncio
import atexit
import logging
import re
import sys
import time
from typing import Any

import httpx
import torch

from datatrove.pipeline.inference.servers import InferenceServer

logger = logging.getLogger(__name__)
sglang_logger = logging.getLogger("sglang")


class SGLangServer(InferenceServer):
    """SGLang inference server implementation."""
    
    async def start_server_task(self, semaphore: asyncio.Semaphore, port: int, offset: int = 0) -> None:
        """Start the SGLang server process."""
        # Check GPU memory, lower mem devices need a bit less KV cache space because the VLM takes additional memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        self.port = self.find_available_port(port, offset)
        # mem_fraction_arg = ["--mem-fraction-static", "0.80"] if gpu_memory < 60 else []

        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_name_or_path,
            # "--mem-fraction-static", "0.82",
            "--chat-template",
            "qwen2-vl",
            "--allow-auto-truncate",
            # "--schedule-conservativeness", "0.1",
            # "--chunked-prefill-size", "8192",
            # "--chat-template",
            # self.chat_template,
            "--context-length", str(self.max_context),
            "--port",
            str(self.port),
            "--log-level-http",
            "warning",
        ]
        # cmd.extend(mem_fraction_arg)

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

        # Shared variables between tasks
        last_running_req, last_queue_req = 0, 0
        server_printed_ready_message = False
        last_semaphore_release = time.time()

        async def process_line(line):
            nonlocal last_running_req, last_queue_req, last_semaphore_release, server_printed_ready_message
            sglang_logger.info(line)

            # if the server hasn't initialized yet, log all the lines to the main logger also, so that the user
            # can see any warnings/errors more easily
            if not server_printed_ready_message:
                logger.info(line)

            if "Detected errors during sampling" in line:
                logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
                sys.exit(1)

            # TODO, need to trace down this issue in sglang itself, but it will otherwise cause the server to lock up
            if "IndexError: list index out of range" in line:
                logger.error("IndexError in model, restarting server")
                if self.process:
                    self.process.terminate()

            if not server_printed_ready_message and "The server is fired up and ready to roll!" in line:
                server_printed_ready_message = True
                last_semaphore_release = time.time()

            match = re.search(r"#running-req: (\d+)", line)
            if match:
                last_running_req = int(match.group(1))

            match = re.search(r"#queue-req: (\d+)", line)
            if match:
                last_queue_req = int(match.group(1))
                logger.info(f"sglang running req: {last_running_req} queue req: {last_queue_req}")

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

        async def timeout_task_impl(semaphore):
            nonlocal last_running_req, last_queue_req, last_semaphore_release
            try:
                while True:
                    await asyncio.sleep(1)
                    if server_printed_ready_message and last_queue_req == 0 and time.time() - last_semaphore_release > 30 and semaphore.locked():
                        semaphore.release()
                        last_semaphore_release = time.time()
                        logger.info("Semaphore released, allowing a worker to proceed.")
            except asyncio.CancelledError:
                pass  # Clean up if the task is cancelled

        # Start tasks to read stdout, stderr, and handle timeout logic
        stdout_task = asyncio.create_task(read_stream(self.process.stdout))
        stderr_task = asyncio.create_task(read_stream(self.process.stderr))
        timeout_task = asyncio.create_task(timeout_task_impl(semaphore))

        try:
            await self.process.wait()
        except asyncio.CancelledError:
            logger.info("Got cancellation request for SGLang server")
            if self.process:
                self.process.terminate()
            raise

        timeout_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, timeout_task, return_exceptions=True)

    async def is_ready(self) -> bool:
        """Check if the SGLang server is ready."""
        if not self.port:
            return False
        url = f"http://localhost:{self.port}/v1/models"
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url)
                return response.status_code == 200
        except Exception:
            return False 