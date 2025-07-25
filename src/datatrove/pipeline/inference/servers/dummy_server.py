import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from datatrove.pipeline.inference.servers import InferenceServer
from loguru import logger


class DummyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            # Read the request body
            content_length = int(self.headers['Content-Length'])
            request_body = self.rfile.read(content_length)
            
            # Parse the request to get basic info
            try:
                request_data = json.loads(request_body.decode('utf-8'))
                # Estimate tokens based on content length (rough approximation)
                prompt_tokens = len(str(request_data)) // 4  # Rough token estimate
                completion_tokens = 100  # Fixed completion tokens for consistency
            except:
                prompt_tokens = 50
                completion_tokens = 100
            
            response_data = {
                "choices": [{
                    "message": {
                        "content": "This is dummy text content for debugging purposes. Page contains sample text to simulate OCR output."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
            # Send response
            response_body = json.dumps(response_data).encode('utf-8')
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        if self.path == "/v1/models":
            # Simple models endpoint for readiness check
            response_data = {
                "object": "list",
                "data": [{"id": "dummy-model", "object": "model"}]
            }
            response_body = json.dumps(response_data).encode('utf-8')
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default HTTP server logging
        pass


class DummyServer(InferenceServer):
    """Dummy inference server for debugging and testing."""
    
    def __init__(self, model_name_or_path: str, model_kwargs: dict | None = None):
        # DummyServer doesn't need chat_template or max_context for its simple functionality
        super().__init__(
            model_name_or_path=model_name_or_path,
            max_context=8192,       # placeholder  
            model_kwargs=model_kwargs
        )
        self.server: HTTPServer = None
        
    async def start_server_task(self, offset: int = 0) -> None:
        """Start the dummy HTTP server in a separate thread."""
        # Find available port for this instance
        self.port = self.find_available_port(offset)
        
        def run_server():
            self.server = HTTPServer(('localhost', self.port), DummyHandler)
            logger.info(f"Dummy server started on port {self.port}")
            self.server.serve_forever()
        
        # Run the server in a separate thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait a bit for the server to start
        await asyncio.sleep(1)
        logger.info("Dummy server is ready!")
        
        # Keep the task alive
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            if self.server:
                self.server.shutdown()
            raise

    async def is_ready(self) -> bool:
        """Check if dummy server is ready (simpler than sglang check)."""
        import httpx
        url = f"http://localhost:{self.port}/v1/models"
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url)
                return response.status_code == 200
        except Exception:
            return False 