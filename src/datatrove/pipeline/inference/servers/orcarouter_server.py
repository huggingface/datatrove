import asyncio
from typing import TYPE_CHECKING

from datatrove.pipeline.inference.servers.base import InferenceServer
from datatrove.utils._import_utils import check_required_dependencies


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig


ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"


class OrcaRouterServer(InferenceServer):
    """Inference server that sends requests to OrcaRouter's OpenAI-compatible endpoint."""

    def __init__(self, config: "InferenceConfig", rank: int):
        """
        Initialize the OrcaRouter server.

        Args:
            config: InferenceConfig containing all server configuration parameters.
                The API key is read from ``config.api_key`` or the ``ORCAROUTER_API_KEY``
                environment variable. ``endpoint_url`` may be overridden to point at a
                self-hosted/on-prem deployment of the OrcaRouter gateway.
            rank: Rank of the server
        """
        super().__init__(config, rank)
        import os

        self.endpoint_url = (config.endpoint_url or ORCAROUTER_BASE_URL).rstrip("/")
        self.api_key = config.api_key or os.getenv("ORCAROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "An API key is required for OrcaRouterServer. Set `api_key` in InferenceConfig "
                "or the ORCAROUTER_API_KEY environment variable."
            )

        # Check if we need the OpenAI client (not needed for localhost HTTP endpoints)
        from urllib.parse import urlparse

        parsed = urlparse(self.endpoint_url)
        is_localhost_http = parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")

        self.client = None
        if not is_localhost_http:
            check_required_dependencies("OrcaRouterServer", ["openai"])
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.endpoint_url,
                timeout=self.config.request_timeout,
            )

    async def start_server(self) -> None:
        """No local server to start - abstract method must be implemented but does nothing."""
        pass

    async def monitor_health(self) -> None:
        """No health check to perform - abstract method must be implemented but does nothing."""
        never_to_be_completed_future = asyncio.Future()
        await never_to_be_completed_future

    async def _wait_until_ready(self, max_attempts: int = 10, delay_sec: float = 1.0) -> None:
        """Wait until the endpoint is ready. Uses shorter delays since we're just checking endpoint availability."""
        await super()._wait_until_ready(max_attempts=max_attempts, delay_sec=delay_sec)

    def get_base_url(self) -> str:
        """Get the base URL for making requests."""
        return self.endpoint_url

    async def is_ready(self) -> bool:
        """
        Check if the endpoint is ready to accept requests.

        The OrcaRouter base URL already includes the OpenAI-compatible ``/v1`` prefix,
        so the model list is probed at ``{base_url}/models``; for custom/localhost base
        URLs that omit the prefix we fall back to ``{base_url}/v1/models``.

        Returns:
            True if the endpoint is ready and responding, False otherwise.
        """
        import httpx

        suffix = "/models" if self.endpoint_url.endswith("/v1") else "/v1/models"
        url = f"{self.endpoint_url}{suffix}"
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url, timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False

    async def _make_request(self, payload: dict) -> dict:
        """
        Make HTTP request to the OrcaRouter endpoint using OpenAI client and return the parsed JSON response.
        Falls back to base implementation for localhost HTTP endpoints.

        Args:
            payload: The request payload to send (should already have model and default params)

        Returns:
            Parsed JSON response dict matching OpenAI API format

        Raises:
            InferenceError: If the request fails
        """
        # For localhost HTTP endpoints, use the faster base implementation
        if self.client is None:
            return await super()._make_request(payload)

        from openai import (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            InternalServerError,
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
        )

        from datatrove.pipeline.inference.run_inference import InferenceError

        # Extract model (required parameter for OpenAI client) and remove from payload
        model = payload.pop("model")

        try:
            if self.config.use_chat:
                response = await self.client.chat.completions.create(model=model, **payload)
                message = response.choices[0].message
                text = message.content or ""
                finish_reason = response.choices[0].finish_reason
            else:
                response = await self.client.completions.create(model=model, **payload)
                text = response.choices[0].text or ""
                finish_reason = response.choices[0].finish_reason

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            # Return in the same format as the base class _make_request
            if self.config.use_chat:
                return {
                    "choices": [
                        {
                            "message": {"content": text},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage,
                }
            else:
                return {
                    "choices": [
                        {
                            "text": text,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage,
                }
        except (AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError) as e:
            # Non-retryable errors: wrap in InferenceError immediately
            # These won't succeed on retry, so fail fast
            raise InferenceError(None, str(e), payload=payload)
        except APIConnectionError as e:
            # Retryable: convert to ConnectionError so retry logic handles it
            raise ConnectionError(str(e)) from e
        except APITimeoutError as e:
            # Retryable: convert to TimeoutError so retry logic handles it
            raise TimeoutError(str(e)) from e
        except (RateLimitError, InternalServerError) as e:
            # Retryable: convert to ConnectionError so retry logic handles it
            raise ConnectionError(str(e)) from e
        except Exception as e:
            # For other exceptions, wrap in InferenceError
            raise InferenceError(None, str(e), payload=payload)
