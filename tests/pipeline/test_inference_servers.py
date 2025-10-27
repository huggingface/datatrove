"""
Unit tests for inference server classes.

Tests the base server classes (InferenceServer, LocalInferenceServer, RemoteInferenceServer)
and the new VLLMRemoteServer implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from datatrove.pipeline.inference.run_inference import InferenceConfig
from datatrove.pipeline.inference.servers.remote_base import RemoteInferenceServer
from datatrove.pipeline.inference.servers.vllm_remote_server import VLLMRemoteServer


class TestRemoteInferenceServerBase:
    """Tests for RemoteInferenceServer base class."""

    def test_extract_port_with_explicit_http_port(self):
        """Test port extraction from HTTP URL with explicit port."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="http://localhost:8000",
        )

        class TestRemoteServer(RemoteInferenceServer):
            pass

        server = TestRemoteServer(config, endpoint="http://localhost:8000")
        assert server.port == 8000

    def test_extract_port_with_explicit_https_port(self):
        """Test port extraction from HTTPS URL with explicit port."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="https://api.service.com:9000",
        )

        class TestRemoteServer(RemoteInferenceServer):
            pass

        server = TestRemoteServer(config, endpoint="https://api.service.com:9000")
        assert server.port == 9000

    def test_extract_port_default_http(self):
        """Test default port (80) for HTTP URLs without explicit port."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="http://service.com",
        )

        class TestRemoteServer(RemoteInferenceServer):
            pass

        server = TestRemoteServer(config, endpoint="http://service.com")
        assert server.port == 80

    def test_extract_port_default_https(self):
        """Test default port (443) for HTTPS URLs without explicit port."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="https://api.service.com",
        )

        class TestRemoteServer(RemoteInferenceServer):
            pass

        server = TestRemoteServer(config, endpoint="https://api.service.com")
        assert server.port == 443

    def test_endpoint_trailing_slash_removed(self):
        """Test that trailing slash is removed from endpoint."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="http://localhost:8000/",
        )

        class TestRemoteServer(RemoteInferenceServer):
            pass

        server = TestRemoteServer(config, endpoint="http://localhost:8000/")
        assert server.endpoint == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_is_ready_success(self):
        """Test is_ready returns True when server responds successfully."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="http://localhost:8000",
        )

        class TestRemoteServer(RemoteInferenceServer):
            pass

        server = TestRemoteServer(config, endpoint="http://localhost:8000")

        # Mock httpx.AsyncClient (httpx is imported inside is_ready() function)
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await server.is_ready()

            assert result is True
            mock_client.get.assert_called_once_with("http://localhost:8000/v1/models", timeout=5.0)

    @pytest.mark.asyncio
    async def test_is_ready_failure_non_200(self):
        """Test is_ready returns False when server returns non-200 status."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="http://localhost:8000",
        )

        class TestRemoteServer(RemoteInferenceServer):
            pass

        server = TestRemoteServer(config, endpoint="http://localhost:8000")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await server.is_ready()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_ready_failure_exception(self):
        """Test is_ready returns False when connection fails."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="http://localhost:8000",
        )

        class TestRemoteServer(RemoteInferenceServer):
            pass

        server = TestRemoteServer(config, endpoint="http://localhost:8000")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await server.is_ready()

            assert result is False

    @pytest.mark.asyncio
    async def test_wait_until_ready_success(self):
        """Test wait_until_ready succeeds when server becomes ready."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="http://localhost:8000",
        )

        class TestRemoteServer(RemoteInferenceServer):
            pass

        server = TestRemoteServer(config, endpoint="http://localhost:8000")

        # Mock is_ready to return True
        server.is_ready = AsyncMock(return_value=True)

        # Should not raise exception
        await server.wait_until_ready(max_attempts=3, delay_sec=0.1)

        # is_ready should have been called once
        server.is_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_until_ready_timeout(self):
        """Test wait_until_ready raises exception on timeout."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="http://localhost:8000",
        )

        class TestRemoteServer(RemoteInferenceServer):
            pass

        server = TestRemoteServer(config, endpoint="http://localhost:8000")

        # Mock is_ready to always return False
        server.is_ready = AsyncMock(return_value=False)

        # Should raise exception after max attempts
        with pytest.raises(Exception, match="did not become ready"):
            await server.wait_until_ready(max_attempts=3, delay_sec=0.01)

        # is_ready should have been called max_attempts times
        assert server.is_ready.call_count == 3

    def test_cancel_is_noop(self):
        """Test that cancel() is a no-op for remote servers."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="http://localhost:8000",
        )

        class TestRemoteServer(RemoteInferenceServer):
            pass

        server = TestRemoteServer(config, endpoint="http://localhost:8000")

        # Should not raise any exception
        server.cancel()


class TestVLLMRemoteServer:
    """Tests for VLLMRemoteServer implementation."""

    def test_init_without_endpoint_raises_error(self):
        """Test that VLLMRemoteServer raises ValueError without external_endpoint."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            # external_endpoint not provided
        )

        with pytest.raises(ValueError, match="external_endpoint is required"):
            VLLMRemoteServer(config)

    def test_init_with_endpoint_succeeds(self):
        """Test successful initialization with external_endpoint."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="http://my-vllm-server.com:8000",
        )

        server = VLLMRemoteServer(config)

        assert server.endpoint == "http://my-vllm-server.com:8000"
        assert server.port == 8000
        assert server.config == config

    def test_port_extracted_correctly(self):
        """Test that port is correctly extracted from endpoint URL."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="http://localhost:9999",
        )

        server = VLLMRemoteServer(config)

        assert server.port == 9999

    def test_https_endpoint(self):
        """Test initialization with HTTPS endpoint."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="https://secure-api.com:443",
        )

        server = VLLMRemoteServer(config)

        assert server.endpoint == "https://secure-api.com:443"
        assert server.port == 443

    def test_default_https_port(self):
        """Test that HTTPS URLs without explicit port default to 443."""
        config = InferenceConfig(
            server_type="vllm-remote",
            model_name_or_path="test-model",
            external_endpoint="https://api.example.com",
        )

        server = VLLMRemoteServer(config)

        assert server.port == 443


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
