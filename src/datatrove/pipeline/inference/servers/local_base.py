import asyncio
import logging
import random
import socket
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from loguru import logger

from datatrove.pipeline.inference.servers.base import InferenceServer


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig


class LocalInferenceServer(InferenceServer):
    """
    Base class for local inference servers that spawn and manage server processes.

    This class handles common functionality for servers that need to:
    - Start local server processes (vLLM, SGLang, etc.)
    - Find and bind to available ports
    - Manage server logs
    - Handle server lifecycle (start, stop, retry)
    """

    def __init__(self, config: "InferenceConfig"):
        super().__init__(config)
        self._server_task: Optional[asyncio.Task] = None

    def find_available_port(self, rank: int = 0) -> int:
        """
        Find an available port for the local server.

        Args:
            rank: Process rank, used to offset the initial port selection

        Returns:
            Available port number

        Raises:
            RuntimeError: If no available port found after max attempts
        """

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
                logger.info(f"Using port {port} for {self.__class__.__name__} server.")
                found_available_port = True
                break
            else:
                logger.debug(f"Port {current_port_to_try} is busy. Trying next port...")

        if not found_available_port:
            raise RuntimeError(
                f"Could not find an available port for server after trying {max_port_scan_attempts} ports, "
                f"starting from {initial_port}."
            )

        return port

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

        import os

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

    @abstractmethod
    async def start_server_task(self) -> None:
        """
        Start the local server process.

        This method should:
        1. Build the command to start the server
        2. Spawn the server process
        3. Monitor server output for errors and readiness
        4. Handle server failures and restarts

        Must be implemented by subclasses (VLLMServer, SGLangServer, etc.)
        """
        pass

    async def is_ready(self) -> bool:
        """
        Check if the local server is ready to accept requests.

        Returns:
            True if server is ready, False otherwise
        """
        import httpx

        url = f"http://localhost:{self.port}/v1/models"
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url, timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False

    async def host_server(self, rank: int = 0, max_retries: int = 5) -> None:
        """
        Host the local server with retry logic.

        Args:
            rank: Process rank identifier
            max_retries: Maximum number of restart attempts

        Raises:
            RuntimeError: If server fails to start after max retries
        """
        retry = 0
        self.rank = rank  # Store rank for log file naming

        while retry < max_retries:
            try:
                # Find available port for this attempt
                port = self.find_available_port(rank)
                self.port = port
                self._server_task = asyncio.create_task(self.start_server_task())
                await self._server_task
                logger.warning(f"{self.__class__.__name__} server task ended")
                retry += 1
            except asyncio.CancelledError:
                logger.info(f"Got cancellation request for {self.__class__.__name__} server")
                raise

        if retry >= max_retries:
            raise RuntimeError(f"Failed to start {self.__class__.__name__} server after {max_retries} retries")

    def cancel(self) -> None:
        """Cancel the local server task."""
        if self._server_task:
            self._server_task.cancel()
