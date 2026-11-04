"""Tests for the HTTP fetch media reader"""

import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from datatrove.data import Media, MediaType
from datatrove.pipeline.media.readers.http_fetch import HTTPFetchReader


class _MediaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/media.bin":
            body = b"media-bytes"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


class TestHTTPFetchReader(unittest.TestCase):
    def test_import_without_requests(self):
        """The module should import without `requests`; the step should only fail when it is instantiated."""
        code = (
            "import sys\n"
            "sys.modules['requests'] = None\n"
            "sys.modules['urllib3'] = None\n"
            "from datatrove.pipeline.media.readers.http_fetch import HTTPFetchReader\n"
            "try:\n"
            "    HTTPFetchReader()\n"
            "except ImportError as e:\n"
            "    assert 'pip install requests' in str(e), e\n"
            "else:\n"
            "    raise AssertionError('expected ImportError')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fetch_media(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MediaHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}/media.bin"

        reader = HTTPFetchReader(max_retries=1)
        content, metadata = reader.fetch_media(Media(id="m1", type=MediaType.DOCUMENT, url=url))

        self.assertEqual(content, b"media-bytes")
        self.assertEqual(metadata["fetched_url"], url)
