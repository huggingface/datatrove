"""Tests for the pooled HTTP client that InferenceServer uses to talk to local servers."""

import asyncio

import pytest

from datatrove.pipeline.inference.run_inference import InferenceConfig
from datatrove.pipeline.inference.servers.base import InferenceServer


class _LocalServer(InferenceServer):
    """An InferenceServer pointed at a test socket server, with no process to manage."""

    def __init__(self, config: InferenceConfig, port: int):
        super().__init__(config, rank=0)
        self._port = port

    async def start_server(self) -> None:
        return None

    async def monitor_health(self) -> None:
        await asyncio.Future()


async def _start_socket_server(handle) -> tuple[asyncio.Server, int]:
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _read_request(reader: asyncio.StreamReader) -> None:
    content_length = 0
    while (line := await reader.readline()) not in (b"\r\n", b""):
        if line.lower().startswith(b"content-length:"):
            content_length = int(line.split(b":")[1])
    await reader.readexactly(content_length)


def _config(**kwargs) -> InferenceConfig:
    return InferenceConfig(server_type="dummy", model_name_or_path="dummy", **kwargs)


def test_dropped_connection_raises_connection_error():
    """A connection closed without a response must be retryable, not reported as a server error."""

    async def drop(reader, writer):
        await _read_request(reader)
        writer.close()

    async def run():
        server, port = await _start_socket_server(drop)
        client = _LocalServer(_config(), port)
        try:
            with pytest.raises(ConnectionError):
                await client._make_request({"messages": []})
        finally:
            await client.server_cleanup()
            server.close()

    asyncio.run(run())


def test_request_timeout_is_applied():
    """A server that accepts the request but never answers must not hang the request."""

    async def hang(reader, writer):
        await _read_request(reader)
        await asyncio.sleep(30)

    async def run():
        server, port = await _start_socket_server(hang)
        client = _LocalServer(_config(request_timeout=0.5), port)
        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            with pytest.raises(asyncio.TimeoutError):
                # the outer limit only stops a hang from blocking the test suite
                await asyncio.wait_for(client._make_request({"messages": []}), timeout=10)
            assert loop.time() - start < 5, "request_timeout was not applied"
        finally:
            await client.server_cleanup()
            server.close()

    asyncio.run(run())


def test_connection_pool_matches_max_concurrent_generations():
    async def run():
        client = _LocalServer(_config(max_concurrent_generations=1234), port=1)
        session = client._get_http_session()
        try:
            assert session.connector.limit == 1234
            assert session.timeout.total is None
        finally:
            await client.server_cleanup()

    asyncio.run(run())
