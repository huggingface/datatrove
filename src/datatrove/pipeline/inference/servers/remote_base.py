import asyncio
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from loguru import logger

from datatrove.pipeline.inference.servers.base import InferenceServer


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig


class RemoteInferenceServer(InferenceServer):
    """
    Base class for remote inference servers that connect to external endpoints.

    This class handles common functionality for servers that:
    - Connect to existing external inference services
    - Do not spawn or manage server processes
    - Perform health checks on remote endpoints
    - Extract connection information from URLs

    Unlike LocalInferenceServer, this class does not start or stop servers,
    it only manages connections to already-running services.
    """

    def __init__(self, config: "InferenceConfig", endpoint: str):
        """
        Initialize remote inference server.

        Args:
            config: InferenceConfig containing server configuration
            endpoint: Full URL of the remote server endpoint
                     (e.g., "http://my-server.com:8000" or "https://api.service.com")

        Raises:
            ValueError: If endpoint URL is invalid
        """
        super().__init__(config)
        self.endpoint = endpoint.rstrip("/")  # Remove trailing slash if present
        self.port = self._extract_port(endpoint)

    def _extract_port(self, endpoint: str) -> int:
        """
        Extract port number from endpoint URL.

        Args:
            endpoint: Full URL of the endpoint

        Returns:
            Port number (explicit from URL, or default for scheme)

        Examples:
            "http://localhost:8000" -> 8000
            "https://api.service.com" -> 443
            "http://service.com" -> 80
        """
        parsed = urlparse(endpoint)

        # If port is explicitly specified in URL, use it
        if parsed.port is not None:
            return parsed.port

        # Otherwise use default port for scheme
        if parsed.scheme == "https":
            return 443
        else:  # http or unspecified
            return 80

    async def is_ready(self) -> bool:
        """
        Check if the remote server is ready to accept requests.

        Performs a health check by calling the /v1/models endpoint.

        Returns:
            True if server responds successfully, False otherwise
        """
        import httpx

        url = f"{self.endpoint}/v1/models"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5.0)
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Remote server health check failed: {e}")
            return False

    async def wait_until_ready(self, max_attempts: int = 10, delay_sec: float = 2.0) -> None:
        """
        Wait until the remote server is ready.

        Unlike local servers which may need time to start up, remote servers
        should typically be already running. This method uses shorter timeouts
        and fewer retries than LocalInferenceServer.

        Args:
            max_attempts: Maximum number of health check attempts (default: 10)
            delay_sec: Delay between attempts in seconds (default: 2.0)

        Raises:
            Exception: If server is not ready after max attempts
        """
        for attempt in range(1, max_attempts + 1):
            try:
                if await self.is_ready():
                    logger.info(f"Remote server at {self.endpoint} is ready.")
                    return
            except Exception:
                pass

            if attempt < max_attempts:
                logger.warning(
                    f"Attempt {attempt}/{max_attempts}: Waiting for remote server at {self.endpoint} to become ready..."
                )
                await asyncio.sleep(delay_sec)

        raise Exception(
            f"Remote server at {self.endpoint} did not become ready after {max_attempts} attempts. "
            f"Please verify the server is running and accessible."
        )

    def cancel(self) -> None:
        """
        Cancel/cleanup for remote server.

        Remote servers don't have local processes to cancel, so this is a no-op.
        Provided for API compatibility with LocalInferenceServer.
        """
        pass
