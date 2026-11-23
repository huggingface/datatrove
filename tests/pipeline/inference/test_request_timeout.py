"""`InferenceConfig.request_timeout` with the built-in HTTP client used by vLLM, SGLang, custom and dummy servers."""

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.inference.servers.endpoint_server import EndpointServer
from datatrove.pipeline.writers import JsonlWriter


class StallingHandler(BaseHTTPRequestHandler):
    """Chat completions endpoint that leaves the first `stall_requests` requests unanswered until `release` is set."""

    stall_requests = 0
    calls = 0
    release = threading.Event()

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).calls += 1
        if type(self).calls <= type(self).stall_requests:
            type(self).release.wait(timeout=30)
            return
        data = json.dumps(
            {
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@contextmanager
def stalling_server(stall_requests):
    StallingHandler.stall_requests = stall_requests
    StallingHandler.calls = 0
    StallingHandler.release = threading.Event()
    http_server = ThreadingHTTPServer(("127.0.0.1", 0), StallingHandler)
    http_server.daemon_threads = True
    threading.Thread(target=http_server.serve_forever, daemon=True).start()
    try:
        yield http_server.server_port
    finally:
        StallingHandler.release.set()
        http_server.shutdown()
        http_server.server_close()


def make_local_endpoint_config(port, request_timeout):
    # a localhost endpoint uses the same built-in HTTP client as the vLLM/SGLang servers
    return InferenceConfig(
        server_type="endpoint",
        model_name_or_path="test-model",
        endpoint_url=f"http://127.0.0.1:{port}",
        request_timeout=request_timeout,
        metric_interval=120,
        rollouts_per_document=1,
    )


def make_ready_server(config):
    server = EndpointServer(config, rank=0)
    server._server_ready.set_result(True)
    return server


async def make_ready_request(config, payload):
    return await make_ready_server(config).make_request(payload)


def test_request_timeout_applies_to_builtin_http_client():
    with stalling_server(stall_requests=1) as port:
        config = make_local_endpoint_config(port, request_timeout=0.5)
        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            # the outer timeout only stops the test from hanging if request_timeout is ignored
            asyncio.run(asyncio.wait_for(make_ready_request(config, {"model": "test-model"}), timeout=10))
        assert time.monotonic() - start < 5


@pytest.mark.parametrize("request_timeout", [None, 5])
def test_request_timeout_does_not_affect_answered_requests(request_timeout):
    with stalling_server(stall_requests=0) as port:
        config = make_local_endpoint_config(port, request_timeout=request_timeout)
        response = asyncio.run(make_ready_request(config, {"model": "test-model"}))
    assert response["choices"][0]["message"]["content"] == "hi"


def test_timed_out_request_is_retried(tmp_path, monkeypatch):
    original_sleep = asyncio.sleep

    async def no_backoff(delay, *args, **kwargs):
        await original_sleep(0)

    async def rollout(document, generate):
        return await generate({"messages": []})

    monkeypatch.setattr(asyncio, "sleep", no_backoff)
    with stalling_server(stall_requests=1) as port:
        config = make_local_endpoint_config(port, request_timeout=0.5)
        runner = InferenceRunner(
            rollout_fn=rollout,
            config=config,
            output_writer=JsonlWriter(str(tmp_path / "output"), output_filename="${rank}.jsonl", compression=None),
        )

        async def send():
            return await runner._send_request(make_ready_server(config), {"messages": []}, asyncio.Semaphore(1))

        result = asyncio.run(asyncio.wait_for(send(), timeout=10))
        assert result.text == "hi"
        assert StallingHandler.calls == 2
