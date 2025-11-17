import asyncio
import json
from urllib.parse import urlparse


async def _raw_post(url: str, json_data: dict, timeout: float | None = None) -> tuple[int, bytes]:
    """
    Very small HTTP/1.1 POST helper using the std-lib socket machinery.

    Args:
        url: The target URL for the POST request
        json_data: Dictionary to be sent as JSON payload
        timeout: Request timeout in seconds (None means no timeout)

    Returns:
        Tuple of (status_code, response_body)
    """
    parsed = urlparse(url)

    # Use raw socket for HTTP URLs (faster, no dependency overhead)
    host, port = parsed.hostname, parsed.port or 80
    path = parsed.path or "/"

    async def _socket_request():
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

    return await asyncio.wait_for(_socket_request(), timeout=timeout)
