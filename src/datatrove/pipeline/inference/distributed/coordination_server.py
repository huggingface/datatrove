"""Coordination server for master node health checks and port sharing."""

import asyncio
import json
from typing import Optional
from loguru import logger

from datatrove.pipeline.inference.distributed.utils import get_master_node_host, is_master_node, get_number_of_nodes


async def _raw_get(url: str) -> tuple[int, bytes]:
    """
    Simple HTTP/1.1 GET helper using asyncio.

    Args:
        url: The target URL for the GET request

    Returns:
        Tuple of (status_code, response_body)
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port or 80
    path = parsed.path or "/"

    reader, writer = await asyncio.open_connection(host, port)
    try:
        request = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n").encode()
        writer.write(request)
        await writer.drain()

        # Status line
        status_line = await reader.readline()
        status_parts = status_line.decode().split(" ", 2)
        status_code = int(status_parts[1]) if len(status_parts) >= 2 else 500

        # Headers (skip until empty line)
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break

        # Body
        body = await reader.read()
        return status_code, body
    finally:
        writer.close()
        await writer.wait_closed()


class CoordinationServer:
    """
    Simple HTTP server for coordinating between master and worker nodes.

    Provides endpoints for:
    - /health/{job_id}: Health check endpoint that waits for request and returns okay
    """

    def __init__(self, port: int, job_id: str):
        """
        Initialize the coordination server.

        Args:
            port: Port to run the coordination server on
            job_id: job ID for health check endpoint
        """
        self.port = port
        self.job_id = job_id
        self._server: Optional[asyncio.Server] = None
        self.master_node_host = get_master_node_host()

    async def __start(self) -> None:
        """Start the coordination server."""

        async def handle_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            try:
                # Read request line
                request_line = await reader.readline()
                if not request_line:
                    return

                # Parse request
                parts = request_line.decode().strip().split()
                if len(parts) < 2:
                    return

                method, path = parts[0], parts[1]

                # Read headers (skip them)
                while True:
                    line = await reader.readline()
                    if line in (b"\r\n", b"\n", b""):
                        break

                # Handle endpoints
                if method == "GET" and path == f"/health/{self.job_id}":
                    # Health check endpoint - wait for request and return okay
                    response_body = json.dumps({"status": "ok", "job_id": self.job_id}).encode()
                    response = (
                        f"HTTP/1.1 200 OK\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Length: {len(response_body)}\r\n"
                        f"Connection: close\r\n\r\n"
                    ).encode()
                    writer.write(response + response_body)
                else:
                    # 404 for other paths
                    response = b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n"
                    writer.write(response)

                await writer.drain()
            except Exception as e:
                logger.debug(f"Error in coordination server: {e}")
            finally:
                writer.close()
                await writer.wait_closed()

        self._server = await asyncio.start_server(handle_request, "0.0.0.0", self.port)
        logger.info(f"Coordination server started on port {self.port} with job_id {self.job_id}")
        return None

    async def __stop(self) -> None:
        """Stop the coordination server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.debug("Coordination server stopped")

    async def __aenter__(self) -> "CoordinationServer":
        """Enter the coordination server."""
        if is_master_node() and get_number_of_nodes() > 1:
            await self.__start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the coordination server."""
        try:
            await asyncio.wait_for(self.__stop(), timeout=10.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    async def master_running(self, timeout: float = 5.0) -> bool:
        """Check if the master node is running. Returns True if the master node is running, False otherwise."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                status_code, _ = await asyncio.wait_for(
                    _raw_get(f"http://{self.master_node_host}:{self.port}/health/{self.job_id}"), timeout=timeout
                )
                if status_code == 200:
                    return True
            except (asyncio.CancelledError):
                raise
            except Exception:
                if attempt == max_retries - 1:
                    return False
                await asyncio.sleep(5.0)
        return False
