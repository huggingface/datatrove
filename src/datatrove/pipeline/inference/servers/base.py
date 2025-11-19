import asyncio
import json
import logging
import os
import random
import socket
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from loguru import logger

from datatrove.pipeline.inference.distributed.utils import is_master_node


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig

# --------------------------------------------------------------------------- #
# Low-level, dependency-free HTTP POST helper (kept from the original file)
# --------------------------------------------------------------------------- #
async def _raw_post(url: str, json_data: dict) -> tuple[int, bytes]:
    """
    Very small HTTP/1.1 POST helper using the std-lib socket machinery.

    Args:
        url: The target URL for the POST request
        json_data: Dictionary to be sent as JSON payload

    Returns:
        Tuple of (status_code, response_body)
    """
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


def _find_available_port(rank: int = 0) -> int:
    def _is_port_available(port: int, host: str = "127.0.0.1") -> bool:
        """Check if a port is available on the given host."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
            return True
        except OSError:  # Catches errors like "address already in use"
            return False

    # Generate a random base port first, then add rank
    base_port = random.randint(3000, 50000)
    initial_port = base_port + rank

    # Ensure the port stays within reasonable bounds
    if initial_port > 65535:
        initial_port = 3000 + (initial_port % 62536)  # Wrap around to keep in valid range
    elif initial_port < 3000:
        initial_port = 3000

    port = initial_port  # Initialize port variable
    found_available_port = False
    max_port_scan_attempts = 200  # Increase scan attempts for better success rate

    for attempt_offset in range(max_port_scan_attempts):
        current_port_to_try = initial_port + attempt_offset

        # Wrap around if we exceed max port
        if current_port_to_try > 65535:
            current_port_to_try = 3000 + (current_port_to_try - 65535)

        if _is_port_available(current_port_to_try):
            port = current_port_to_try
            found_available_port = True
            break
        else:
            logger.debug(f"Port {current_port_to_try} is busy. Trying next port...")

    if not found_available_port:
        raise asyncio.CancelledError(
            f"Could not find an available port for server after trying {max_port_scan_attempts} ports, "
            f"starting from {initial_port}."
        )

    return port


class InferenceServer(ABC):
    """Abstract base class for inference servers."""

    _requires_dependencies = ["httpx"]

    def __init__(self, config: "InferenceConfig", rank: int):
        self.config = config

        # Server state - using Future instead
        # On any server error, we reset the future to allow for new start attempt.
        # If the server either fails too many times to boot up or it's detected as not possible to start,
        # the future will be set to an exception. Once in this state, the server will never be started again.
        self._server_ready: asyncio.Future = asyncio.Future()
        self._server_task: Optional[asyncio.Task] = None
        self._port = None
        self._server_process: Optional[asyncio.subprocess.Process] = None
        self._server_start_lock = asyncio.Lock()
        # Monitoring task for local server health check
        self._server_monitoring_task: Optional[asyncio.Task] = None
        self._server_logger: Optional[logging.Logger] = None
        # Auto-restart task for server failure recovery
        self._auto_restart_task: Optional[asyncio.Task] = None
        self._is_master = is_master_node()
        self._rank = rank

    def _get_log_file_path(self, rank: int) -> str | None:
        """Get the log file path for a given rank, or None if logging is disabled."""
        if self.config.server_log_folder is None:
            return None
        return f"{self.config.server_log_folder}/server_rank_{rank}.log"

    def _should_log_to_file(self) -> bool:
        """Check if server output should be logged to file."""
        return self.config.server_log_folder is not None

    def _create_server_logger(self, rank: int):
        """Create a dedicated logger for server output that writes to file."""
        if not self._should_log_to_file():
            return None

        log_file_path = self._get_log_file_path(rank)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Create a dedicated logger for this server instance
        server_logger = logging.getLogger(f"{self.__class__.__name__}_rank_{rank:05d}")
        server_logger.setLevel(logging.INFO)

        # Remove any existing handlers to avoid duplicates
        server_logger.handlers.clear()

        # Create file handler
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(logging.INFO)

        # Create a simple formatter (just the message, no timestamp since server provides its own)
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)

        server_logger.addHandler(file_handler)
        server_logger.propagate = False  # Don't propagate to root logger

        return server_logger

    def server_logger(self) -> Optional[logging.Logger]:
        """Simple cached getter for the server logger."""
        if self._server_logger is None:
            self._server_logger = self._create_server_logger(self._rank)
        return self._server_logger

    @abstractmethod
    async def start_server_task(self) -> asyncio.subprocess.Process | None:
        """Start the server process and return the process object, or None if there is no process to start, if any exception is raised,
        the server will be marked as unable to start and the auto-restart task will be cancelled."""
        pass

    async def is_ready(self) -> bool:
        """Check if the server is ready to accept requests, returns True if the server is ready, False otherwise."""
        import httpx

        url = f"http://localhost:{self._port}/v1/models"
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url, timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False

    async def __aenter__(self):
        """Enter the context manager, starting the auto-restart task and returning the server object and whether this is the master node."""
        self._auto_restart_task = asyncio.create_task(self._server_auto_restart())
        # This is important as we need workers to be barriered by the master node, for the coordination server to be ready.
        # This is very meh in terms of responsbility, as we should pro
        if not self._is_master:
            await self._server_ready
        return self, self._is_master

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, cancelling the auto-restart task and cleaning up the server."""
        # We must first cancel the auto restart task to avoid race conditions
        if self._auto_restart_task:
            try:
                self._auto_restart_task.cancel()
                await asyncio.wait_for(self._auto_restart_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        await self.server_cleanup()

    async def send_request(self, endpoint: str, request: dict) -> tuple[int, bytes]:
        """Send a request to the server, returns the status code and the response body. Raises an exception if the server is unable to start.
        Some of them are retryable, some are not. Thos that are not are:
        - asyncio.CancelledError: if the server is unable to start
        """
        url = f"http://localhost:{self._port}{endpoint}"
        # Wait for server to be ready (will raise if exception was set). Hopefuly this is "atomic" == doesn't give up
        # the execution context when the the future is already set. This is "safe way" to ensure this really happens.
        if not self._server_ready.done() or self._server_ready.exception() is not None:
            await self._server_ready

        return await _raw_post(url, request)

    async def _server_auto_restart(self) -> None:
        while True:
            try:
                if not self._server_ready.done():
                    await self.start_server()
            except asyncio.CancelledError as e:
                # Server failed permanently after max retries - stop auto-restart
                logger.error(f"Server auto-restart stopped: {e}")
                break
            except Exception as e:
                logger.warning(f"Temporary server start failure: {e}")
                pass
            await asyncio.sleep(10)

    async def wait_until_ready(self, max_attempts: int = 300, delay_sec: float = 5.0) -> None:
        """Wait until the server is ready."""
        for attempt in range(1, max_attempts + 1):
            try:
                # Check if server process is still running
                if self._server_process is not None and self._server_process.returncode is not None:
                    raise asyncio.CancelledError(
                        f"{self.__class__.__name__} server process terminated unexpectedly "
                        f"with return code {self._server_process.returncode}"
                    )
                
                if await self.is_ready():
                    logger.info(f"{self.__class__.__name__} server is ready.")
                    return
            except asyncio.CancelledError:
                # Re-raise asyncio.CancelledError (process termination) immediately
                raise
            except Exception:
                pass
            logger.warning(f"Attempt {attempt}: Please wait for {self.__class__.__name__} server to become ready...")
            await asyncio.sleep(delay_sec)

        raise Exception(f"{self.__class__.__name__} server did not become ready after waiting.")

    async def server_cleanup(self) -> None:
        # Reset the future for next start attempt, only if the server were previously ready.
        if self._server_ready.done() and self._server_ready.exception() is None:
            self._server_ready = asyncio.Future()
        
        if self._server_process:
            try:
                self._server_process.terminate()
                await asyncio.wait_for(self._server_process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                self._server_process.kill()
            except ProcessLookupError:
                pass
            self._server_process = None

        if self._server_monitoring_task:
            try:
                self._server_monitoring_task.cancel()
                await asyncio.wait_for(self._server_monitoring_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._server_monitoring_task = None

    @abstractmethod
    async def monitor_health(self):
        """Monitor the health of the server, exiting or raising an exception if the server is not healthy"""
        pass

    async def _monitor_server(self):
        try:
            await self.monitor_health()
        except RuntimeError as e:
            logger.error(f"Server health check failed: {e}")
            # THe monitor_health can signal non-recoverable errors, in which case we should stop auto-restart.
            self._server_ready.set_exception(asyncio.CancelledError("Non-recoverable server error"))
        except Exception as e:
            logger.error(f"Server health check failed: {e}")
            pass

        # Atomic update - reset future on health check failure, only if we were previously ready.
        if self._server_ready.done() and self._server_ready.exception() is None:
            self._server_ready = asyncio.Future()
        # We don't kill server here as weird things could happen as we can't guarantee atomicity of this call,
        # we just kill it during start server call


    async def start_server(self, max_retries: int = 1) -> None:
        # We shouldn't need a lock as there is just single process who can run start_sever at time, but just to be sure.
        async with self._server_start_lock:
            # Check if server is already ready after acquiring lock
            if self._server_ready.done() and self._server_ready.exception() is None:
                return
            # If the server is already failed, we don't try to start it again.
            elif self._server_ready.done() and self._server_ready.exception() is not None:
                await self._server_ready

            retry = 0
            while retry < max_retries:
                # Cleanup the server if it is already running
                await self.server_cleanup()

                # Find available port for this attempt
                port = _find_available_port(self._rank)
                self._port = port
                try:
                    self._server_process = await asyncio.wait_for(self.start_server_task(), timeout=60.0)
                    self._server_monitoring_task = asyncio.create_task(self._monitor_server())
                    # We have no way to know if the server is ready from child process
                    if self._is_master:
                        await self.wait_until_ready()
                    # Signal success
                    if not self._server_ready.done():
                        self._server_ready.set_result(True)
                    return
                except (asyncio.TimeoutError):
                    logger.warning(f"Server start attempt {retry + 1}/{max_retries} timed out")
                    pass
                except Exception as e:
                    logger.warning(f"Server start attempt {retry + 1}/{max_retries} failed: {e}")
                    pass
                retry += 1

            # Failed after all retries - signal error to all waiters
            error_msg = f"Failed to start {self.__class__.__name__} server after {max_retries} retries"
            if not self._server_ready.done():
                self._server_ready.set_exception(asyncio.CancelledError(error_msg))
            raise asyncio.CancelledError(error_msg)


