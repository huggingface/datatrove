import asyncio
import logging
import random
import socket
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from loguru import logger


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig


class InferenceServer(ABC):
    """Abstract base class for inference servers."""

    _requires_dependencies = ["httpx"]

    def __init__(self, config: "InferenceConfig"):
        self.config = config
        self._server_task: Optional[asyncio.Task] = None
        self.port = None

    def find_available_port(self, rank: int = 0) -> int:
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
        """Start the server process."""
        pass

    async def is_ready(self) -> bool:
        """Check if the server is ready to accept requests."""
        import httpx

        url = f"http://localhost:{self.port}/v1/models"
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url, timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False

    async def host_server(self, rank: int = 0, max_retries: int = 5) -> None:
        """Host the server with retry logic."""
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

    async def wait_until_ready(self, max_attempts: int = 300, delay_sec: float = 5.0) -> None:
        """Wait until the server is ready."""
        for attempt in range(1, max_attempts + 1):
            try:
                if await self.is_ready():
                    logger.info(f"{self.__class__.__name__} server is ready.")
                    return
            except Exception:
                pass
            logger.warning(f"Attempt {attempt}: Please wait for {self.__class__.__name__} server to become ready...")
            await asyncio.sleep(delay_sec)

        raise Exception(f"{self.__class__.__name__} server did not become ready after waiting.")

    def cancel(self) -> None:
        """Cancel the server task."""
        if self._server_task:
            self._server_task.cancel()
