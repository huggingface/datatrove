import asyncio
import logging
from abc import ABC, abstractmethod
import socket
from typing import Any, Optional
from loguru import logger
import random

class InferenceServer(ABC):
    """Abstract base class for inference servers."""
    
    def __init__(self, model_name_or_path: str, chat_template: str, max_context: int, model_kwargs: Optional[dict] = None):
        self.model_name_or_path = model_name_or_path
        self.chat_template = chat_template
        self.max_context = max_context
        self.process: Optional[asyncio.subprocess.Process] = None
        self._server_task: Optional[asyncio.Task] = None
        self.port = None
        self.model_kwargs = model_kwargs or {}

    def find_available_port(self, offset: int = 0) -> int:
        def _is_port_available(port: int, host: str = "127.0.0.1") -> bool:
            """Check if a port is available on the given host."""
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((host, port))
                return True
            except OSError:  # Catches errors like "address already in use"
                return False

        # Generate a random base port first, then add offset
        base_port = random.randint(3000, 50000)
        initial_port = base_port + offset
        
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

        
    @abstractmethod
    async def start_server_task(self, semaphore: asyncio.Semaphore, offset: int = 0) -> None:
        """Start the server process."""
        pass
    
    @abstractmethod
    async def is_ready(self) -> bool:
        """Check if the server is ready to accept requests."""
        pass
    
    async def host_server(self, semaphore: asyncio.Semaphore, offset: int = 0, max_retries: int = 5) -> None:
        """Host the server with retry logic."""
        retry = 0
        
        while retry < max_retries:
            try:
                # Find available port for this attempt
                if not self.port:
                    self.port = self.find_available_port(offset)
                
                await self.start_server_task(semaphore, self.port, offset)
                logger.warning(f"{self.__class__.__name__} server task ended")
                retry += 1
                # Reset port for next retry attempt
                self.port = None
            except asyncio.CancelledError:
                logger.info(f"Got cancellation request for {self.__class__.__name__} server")
                if self.process:
                    self.process.terminate()
                raise
        
        if retry >= max_retries:
            logger.error(f"Ended up starting the {self.__class__.__name__} server more than {retry} times, cancelling pipeline")
            raise RuntimeError(f"Failed to start {self.__class__.__name__} server after {max_retries} retries")
    
    async def wait_until_ready(self, max_attempts: int = 300, delay_sec: float = 1.0) -> None:
        """Wait until the server is ready."""
        for attempt in range(1, max_attempts + 1):
            try:
                if await self.is_ready():
                    logger.info(f"{self.__class__.__name__} server is ready.")
                    return
            except Exception as e:
                pass
            logger.warning(f"Attempt {attempt}: Please wait for {self.__class__.__name__} server to become ready...")
            await asyncio.sleep(delay_sec)
        
        raise Exception(f"{self.__class__.__name__} server did not become ready after waiting.")
    
    def cancel(self) -> None:
        """Cancel the server task."""
        if self._server_task:
            self._server_task.cancel() 