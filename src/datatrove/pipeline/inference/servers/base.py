import asyncio
import json
import logging
import os
import random
import socket
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from loguru import logger

from datatrove.pipeline.inference.distributed.utils import get_node_rank, is_master_node
from datatrove.pipeline.inference.types import ServerError


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
        raise ServerError(
            f"Could not find an available port for server after trying {max_port_scan_attempts} ports, "
            f"starting from {initial_port}."
        )

    return port


class InferenceServer(ABC):
    """
    Abstract base class for inference servers.

    This class provides the infrastructure for managing inference server processes,
    including automatic startup, health monitoring, and
    request handling. Subclasses must implement the abstract methods to define
    server-specific behavior.

    The server lifecycle is managed through an asyncio Future that tracks the
    server's readiness state. On errors, the future can be reset to allow retry
    attempts, or set to an exception if the server cannot be started.

    The responsiblity for the lifetime management is delegated to this server, no
    other code should be responsible for the server's lifetime.

    Attributes:
        _requires_dependencies: List of required dependencies for the server.
        config: The inference configuration object.
        _rank: The rank/ID of this server instance.
        _is_master: Whether this is the master node in a distributed setup.
    """

    _requires_dependencies = ["httpx"]

    def __init__(self, config: "InferenceConfig", rank: int):
        """
        Initialize the inference server.

        Args:
            config: The inference configuration containing server settings.
            rank: The rank/ID of this server instance, used for port selection
                and logging.
        """
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
        # Background task for starting the server (so we can start preprocessing)
        self._bg_start_server_task: Optional[asyncio.Task] = None
        self._is_master = is_master_node()
        self._rank = rank
        self._node_rank = get_node_rank()

    # --------------------------------------------------------------------------- #
    # Logging helper methods
    # --------------------------------------------------------------------------- #

    def _get_log_file_path(self) -> str | None:
        """
        Get the log file path for a given rank.

        Returns:
            The log file path string, or None if logging is disabled.
        """
        if self.config.server_log_folder is None:
            return None
        # TODO
        return f"{self.config.server_log_folder}/server_rank_{self._rank:05d}_node_{self._node_rank}.log"

    def _should_log_to_file(self) -> bool:
        """
        Check if server output should be logged to file.

        Returns:
            True if server logging is enabled, False otherwise.
        """
        return self.config.server_log_folder is not None

    def _create_server_logger(self) -> Optional[logging.Logger]:
        """
        Create a dedicated logger for server output that writes to file.

        The logger is configured to write only the message content (no timestamps)
        since the server process typically provides its own timestamps.

        Returns:
            A configured Logger instance, or None if logging is disabled.
        """
        if not self._should_log_to_file():
            return None

        log_file_path = self._get_log_file_path()
        if log_file_path is None:
            return None

        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Create a dedicated logger for this server instance
        server_logger = logging.getLogger(f"{self.__class__.__name__}_rank_{self._rank:05d}_node_{self._node_rank}")
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
        """
        Get or create the server logger instance.

        This is a cached getter that creates the logger on first access if
        logging is enabled.

        Returns:
            The server logger instance, or None if logging is disabled.
        """
        if self._server_logger is None:
            self._server_logger = self._create_server_logger()
        return self._server_logger

    # --------------------------------------------------------------------------- #
    # Abstract methods (must be implemented by subclasses)
    # --------------------------------------------------------------------------- #

    @abstractmethod
    async def start_server(self) -> asyncio.subprocess.Process | None:
        """
        Start the server process and return the process object.

        This method is called by the base class to start the actual server
        process. Subclasses must implement this to launch their specific
        server implementation.

        Returns:
            The asyncio subprocess Process object if a process was started,
            or None if there is no process to start (e.g., external server).

        Raises:
            Exception: Any exception raised here will mark the server as unable
                to start.
        """
        pass

    @abstractmethod
    async def monitor_health(self) -> None:
        """
        Monitor the health of the server.

        This method should continuously monitor the local server's health status.
        It should exit or raise an exception if the server is not healthy.
        The method is called in a background task and should run until the
        server becomes unhealthy or is stopped.

        Raises:
            Exception: If the server encounters a non-recoverable error. (we now treat all errors as non-recoverable)
        """
        pass

    # --------------------------------------------------------------------------- #
    # Public API methods
    # --------------------------------------------------------------------------- #

    def get_base_url(self) -> str:
        """Get the base URL for making requests. Defaults to localhost with port."""
        return f"http://localhost:{self._port}"

    async def is_ready(self) -> bool:
        """
        Check if the server is ready to accept requests.

        Performs a health check by making a GET request to the server's
        /v1/models endpoint. Returns True if the server responds with
        a 200 status code.

        Returns:
            True if the server is ready and responding, False otherwise.
        """
        import httpx

        url = f"{self.get_base_url()}/v1/models"
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url, timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False

    async def make_request(self, payload: dict) -> dict:
        """
        Make a request to the server.

        Args:
            payload: The request payload to send (should already have model and default params)

        Returns:
            Parsed JSON response dict

        Raises:
            ServerError: If the server is unable to start or has
                permanently failed.
        """
        # Wait for server to be ready (will raise if exception was set). Hopefuly this is "atomic" == doesn't give up
        # the execution context when the the future is already set. This is "safe way" to ensure this really happens.
        if not self._server_ready.done() or self._server_ready.exception() is not None:
            await self._server_ready

        return await self._make_request(payload=payload)

    # --------------------------------------------------------------------------- #
    # Context manager methods
    # --------------------------------------------------------------------------- #

    async def __aenter__(self):
        """
        Enter the context manager.

        Starts the start task in the background.

        Returns:
            - self: The server instance.
        """
        # We start this task no matter what, as we need a task that launches the server in background so that we can early return
        # for master node.
        self._bg_start_server_task = asyncio.create_task(self.bg_start_server())
        if self._is_master:
            return self
        else:
            # we are a worker and need to block until everything is done and fully finished
            await self._bg_start_server_task

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.

        Cancels the background start task and performs server cleanup. Ensures
        proper shutdown of all server-related tasks and processes.

        Args:
            exc_type: Exception type, if any.
            exc_val: Exception value, if any.
            exc_tb: Exception traceback, if any.
        """
        # We must first cancel the start task to avoid race conditions
        if self._bg_start_server_task:
            try:
                self._bg_start_server_task.cancel()
                await asyncio.wait_for(self._bg_start_server_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        await self.server_cleanup()

    # --------------------------------------------------------------------------- #
    # Server lifecycle methods
    # --------------------------------------------------------------------------- #

    async def bg_start_server(self, max_retries: int = 1) -> None:
        """
        Start the server with retry logic.

        Attempts to start the server process, with optional retries on failure.
        Uses a lock to ensure only one start attempt happens at a time. If the
        server is already ready, returns immediately. If it has permanently
        failed, re-raises the exception.

        Args:
            max_retries: Maximum number of start attempts. Default is 1.

        Raises:
            ServerError: If the server fails to start after all
                retry attempts, or if it has previously failed permanently.
        """
        # We shouldn't need a lock as there is just single process who can run start_sever at time, but just to be sure.
        async with self._server_start_lock:
            try:
                # Check if server is already ready after acquiring lock
                if self._server_ready.done():
                    return

                for retry in range(max_retries + 1):
                    # Cleanup the server if it is already running
                    await self.server_cleanup()

                    # Find available port for this attempt
                    self._port = _find_available_port(self._rank)
                    try:
                        self._server_process = await asyncio.wait_for(self.start_server(), timeout=60.0)
                        self._server_monitoring_task = asyncio.create_task(self._monitor_server())
                        # We have no way to know if the server is ready from child process
                        if self._is_master:
                            await self._wait_until_ready()
                            # Signal success
                            if not self._server_ready.done():
                                self._server_ready.set_result(True)
                        else:
                            # we need to block the worker
                            await self._server_monitoring_task
                        # we are done
                        return
                    except asyncio.CancelledError:
                        raise
                    except asyncio.TimeoutError:
                        logger.warning(f"Server start attempt {retry + 1}/{max_retries} timed out")
                    except Exception as e:
                        logger.warning(f"Server start attempt {retry + 1}/{max_retries} failed: {e}")

                # Failed after all retries - signal error to all waiters
                error_msg = f"Failed to start {self.__class__.__name__} server after {max_retries} retries"
                raise ServerError(error_msg)
            except Exception as e:
                if not self._server_ready.done():
                    self._server_ready.set_exception(ServerError(e))

    async def server_cleanup(self) -> None:
        """
        Clean up server resources.

        Resets the server ready future (if it was previously ready), terminates
        the server process, and cancels the monitoring task. This method is
        idempotent and safe to call multiple times.
        """
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

    # --------------------------------------------------------------------------- #
    # Internal/private methods
    # --------------------------------------------------------------------------- #
    async def _monitor_server(self) -> None:
        """
        Background task that monitors server health.

        Wraps the abstract monitor_health() method and handles exceptions.
        On health check failure, resets the server ready future to allow start attempts.
        If the monitor_health raises an exception instead of returning,
        the server is marked as permanently failed and we will not attempt to start it again.

        Note:
            This is an internal method that should not be called directly.
            It's started automatically when the server process starts.
        """
        server_error_msg = "Server unexpectedly terminated"
        try:
            await self.monitor_health()
        except Exception as e:
            server_error_msg = e

        # We no longer can recover from server error, if it was previously running, we thus
        # set the server ready exception to kill coming requests.
        if self._server_ready.done() and self._server_ready.exception() is None:
            self._server_ready = asyncio.Future()
            self._server_ready.set_exception(ServerError(server_error_msg))

        # Server failed during startup -> don't do anything let the bg_start_server handle it

    async def _wait_until_ready(self, max_attempts: int = 300, delay_sec: float = 5.0) -> None:
        """
        Wait until the server is ready to accept requests.

        Polls the server's readiness status at regular intervals until it
        becomes ready or the maximum number of attempts is reached. Also
        checks if the server process has terminated unexpectedly.

        Args:
            max_attempts: Maximum number of readiness check attempts.
                Default is 300.
            delay_sec: Delay in seconds between readiness checks.
                Default is 5.0 seconds.

        Raises:
            ServerError: If the server process terminates
                unexpectedly.
            Exception: If the server does not become ready after the
                maximum number of attempts.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                # Check if server process is still running
                if self._server_process is not None and self._server_process.returncode is not None:
                    raise ServerError(
                        f"{self.__class__.__name__} server process terminated unexpectedly "
                        f"with return code {self._server_process.returncode}"
                    )

                if await self.is_ready():
                    logger.info(f"{self.__class__.__name__} server is ready.")
                    return
            except ServerError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            logger.warning(f"Attempt {attempt}: Please wait for {self.__class__.__name__} server to become ready...")
            await asyncio.sleep(delay_sec)

        raise ServerError(f"{self.__class__.__name__} server did not become ready after waiting.")

    def cancel(self) -> None:
        """Cancel the server task."""
        if self._server_task:
            self._server_task.cancel()

    async def _make_request(self, payload: dict) -> dict:
        """
        Make HTTP request to the server and return the parsed JSON response.

        Args:
            payload: The request payload to send (should already have model and default params)

        Returns:
            Parsed JSON response dict

        Raises:
            InferenceError: If the request fails
        """
        from datatrove.pipeline.inference.run_inference import InferenceError

        # Choose endpoint based on use_chat setting
        if self.config.use_chat:
            endpoint = "/v1/chat/completions"
        else:
            endpoint = "/v1/completions"

        url = f"{self.get_base_url()}{endpoint}"
        status, body = await _raw_post(url, json_data=payload)

        if status == 400:
            raise InferenceError(None, f"Got BadRequestError from server: {body.decode()}", payload=payload)
        elif status == 500:
            raise ServerError(f"Got InternalServerError from server: {body.decode()}")
        elif status != 200:
            raise InferenceError(None, f"Error http status {status}", payload=payload)

        return json.loads(body)
