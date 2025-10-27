import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from loguru import logger


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig


class InferenceServer(ABC):
    """
    Abstract base class for all inference servers.

    This minimal interface defines the common contract that all inference servers
    (both local and remote) must implement. Concrete functionality is provided by:
    - LocalInferenceServer: For servers that spawn local processes
    - RemoteInferenceServer: For servers that connect to external endpoints
    """

    _requires_dependencies = ["httpx"]

    def __init__(self, config: "InferenceConfig"):
        """
        Initialize inference server.

        Args:
            config: InferenceConfig containing server configuration parameters
        """
        self.config = config
        self.port: Optional[int] = None

    @abstractmethod
    async def is_ready(self) -> bool:
        """
        Check if the server is ready to accept requests.

        Returns:
            True if server is ready, False otherwise

        Must be implemented by subclasses to provide appropriate
        readiness checks for their server type.
        """
        pass

    async def wait_until_ready(self, max_attempts: int = 300, delay_sec: float = 5.0) -> None:
        """
        Wait until the server is ready to accept requests.

        Default implementation with retry logic. Subclasses may override
        to provide custom timeout/retry behavior.

        Args:
            max_attempts: Maximum number of readiness check attempts
            delay_sec: Delay between attempts in seconds

        Raises:
            Exception: If server is not ready after max attempts
        """
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
        """
        Cancel/cleanup the server.

        Default no-op implementation. Subclasses should override if they
        need to perform cleanup (e.g., stopping processes, closing connections).
        """
        pass
