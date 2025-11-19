import argparse
import asyncio
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING

import torch.distributed as dist
from loguru import logger

from datatrove.pipeline.inference.servers import InferenceServer


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig


class DummyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            # Read the request body
            content_length = int(self.headers["Content-Length"])
            request_body = self.rfile.read(content_length)

            # Parse the request to get basic info
            try:
                request_data = json.loads(request_body.decode("utf-8"))
                # Estimate tokens based on content length (rough approximation)
                prompt_tokens = len(str(request_data)) // 4  # Rough token estimate
                completion_tokens = 100  # Fixed completion tokens for consistency
            except Exception:
                prompt_tokens = 50
                completion_tokens = 100

            response_data = {
                "choices": [
                    {
                        "message": {
                            "content": "This is dummy text content for debugging purposes. Page contains sample text to simulate OCR output."
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

            # Send response
            response_body = json.dumps(response_data).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            # Simple models endpoint for readiness check
            response_data = {"object": "list", "data": [{"id": "dummy-model", "object": "model"}]}
            response_body = json.dumps(response_data).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default HTTP server logging
        pass


class DummyServer(InferenceServer):
    """Dummy inference server for debugging and testing."""

    def __init__(self, config: "InferenceConfig", rank: int):
        """
        Initialize Dummy server.

        Args:
            config: InferenceConfig containing all server configuration parameters
            rank: Rank of the server
        """
        super().__init__(config, rank)

    async def start_server_task(self) -> asyncio.subprocess.Process | None:
        """Start the dummy HTTP server."""
        world_size = self.config.tp * self.config.dp * self.config.pp
        print(f"World size: {world_size}")

        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(world_size),
            os.path.abspath(__file__),
            "--port",
            str(self._port),
        ]
        logger.info(f"Starting distributed Dummy server with: {' '.join(cmd)}")
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )


    async def monitor_health(self):
        if self._server_process:
            async def read_stream(stream):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line = line.decode("utf-8").rstrip()
                    if "Dummy server started on port" in line:
                        logger.info(line)

            stdout_task = asyncio.create_task(read_stream(self._server_process.stdout))
            stderr_task = asyncio.create_task(read_stream(self._server_process.stderr))
            try:
                await asyncio.gather(stdout_task, stderr_task, self._server_process.wait())
            finally:
                stdout_task.cancel()
                stderr_task.cancel()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    logger.info(f"Args: {args}")

    if "RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        dist.barrier()
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    if rank == 0:
            server = HTTPServer(("localhost", args.port), DummyHandler)
            server.serve_forever()
    else:
        while True:
            time.sleep(1)
