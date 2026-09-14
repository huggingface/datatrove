import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.inference.servers.endpoint_server import EndpointServer
from datatrove.pipeline.writers import JsonlWriter


@pytest.mark.parametrize(
    "message, finish_reason, expected_reasoning",
    [
        ({"content": "answer", "reasoning_content": "provider reasoning"}, "stop", "provider reasoning"),
        ({"content": "answer", "reasoning": "reasoning trace"}, "stop", "reasoning trace"),
        ({"content": None, "reasoning_content": "still thinking"}, "length", "still thinking"),
        ({"content": None, "reasoning": "still thinking"}, "length", "still thinking"),
        ({"content": "answer"}, "stop", ""),
        ({"content": "answer", "reasoning": None, "reasoning_content": None}, "stop", ""),
        (
            {"content": "answer", "reasoning": "primary", "reasoning_content": "alternate"},
            "stop",
            "primary",
        ),
        ({"content": "answer", "reasoning": "", "reasoning_content": "fallback"}, "stop", "fallback"),
        ({"content": "answer", "reasoning": None, "reasoning_content": "fallback"}, "stop", "fallback"),
    ],
)
def test_endpoint_server_preserves_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    message: dict,
    finish_reason: str,
    expected_reasoning: str,
) -> None:
    """Normalize SDK reasoning fields to a single reasoning field and preserve it in the runner."""
    openai = pytest.importorskip("openai")
    from openai.types.chat import ChatCompletion

    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    response = ChatCompletion.model_validate(
        {
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish_reason}],
            "usage": usage,
        }
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    monkeypatch.setattr(openai, "AsyncOpenAI", MagicMock(return_value=client))

    async def run_test() -> None:
        config = InferenceConfig(
            server_type="endpoint",
            model_name_or_path="test-model",
            endpoint_url="https://example.com/v1",
            api_key="test-key",
            max_concurrent_generations=1,
        )
        server = EndpointServer(config, rank=0)
        server._server_ready.set_result(None)
        payload = {"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}
        normalized = await server.make_request(payload.copy())
        normalized_message = normalized["choices"][0]["message"]
        assert normalized_message == {"content": message["content"] or "", "reasoning": expected_reasoning}

        runner = InferenceRunner(
            rollout_fn=lambda document, generate: generate({}),
            config=config,
            output_writer=JsonlWriter(str(tmp_path), output_filename="${rank}.jsonl", compression=None),
        )
        result = await runner._send_request(server, payload.copy(), asyncio.Semaphore(1))
        assert result.text == (message["content"] or "")
        assert result.reasoning == expected_reasoning
        assert result.finish_reason == finish_reason
        assert result.usage == usage

    asyncio.run(run_test())
