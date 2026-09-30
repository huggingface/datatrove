import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.inference.types import InferenceResult
from datatrove.pipeline.writers import JsonlWriter


def make_runner(tmp_path: Path, use_chat: bool = True, cache: bool = False) -> InferenceRunner:
    """Build a runner that returns generation results directly from its rollout."""
    return InferenceRunner(
        rollout_fn=lambda document, generate: generate({"messages": [{"role": "user", "content": document.text}]}),
        config=InferenceConfig(
            server_type="dummy",
            model_name_or_path="test-model",
            use_chat=use_chat,
            max_concurrent_generations=1,
        ),
        output_writer=JsonlWriter(
            str(tmp_path / "output"), output_filename="${rank}_${chunk_index}.jsonl", compression=None
        ),
        checkpoints_local_dir=str(tmp_path / "checkpoints") if cache else None,
    )


def make_server(use_chat: bool = True) -> SimpleNamespace:
    """Build a server stub with a deterministic chat or completion response."""
    choice = {"message": {"content": "answer"}} if use_chat else {"text": "answer"}
    return SimpleNamespace(
        make_request=AsyncMock(return_value={"choices": [{**choice, "finish_reason": "stop"}], "usage": {}})
    )


def test_inference_result_model_is_optional() -> None:
    """Existing positional constructors keep their meaning when model is omitted."""
    result = InferenceResult("answer", "stop", {}, "reasoning trace")
    assert result.reasoning == "reasoning trace"
    assert result.model == ""


@pytest.mark.parametrize("use_chat", [True, False])
def test_request_result_includes_configured_model(tmp_path: Path, use_chat: bool) -> None:
    """Chat and completion results record the configured request model."""

    async def run_test() -> None:
        runner = make_runner(tmp_path, use_chat=use_chat)
        server = make_server(use_chat)
        result = await runner._send_request(server, {}, asyncio.Semaphore(1))
        assert result.model == "test-model"
        assert result.model == server.make_request.call_args.args[0]["model"]

    asyncio.run(run_test())


def test_request_cache_preserves_model_on_replay(tmp_path: Path) -> None:
    """Persist model information and restore it from a reopened request cache."""

    async def run_test() -> None:
        runner = make_runner(tmp_path, cache=True)
        server = make_server()
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        await runner.request_cache.initialize(rank=0)
        try:
            result = await runner._cached_request(
                payload.copy(), asyncio.Semaphore(1), server, doc_id="doc-1", rollout_idx=0, chunk_index=0
            )
            assert result.model == "test-model"
        finally:
            await runner.request_cache.close()

        resumed_runner = make_runner(tmp_path, cache=True)
        await resumed_runner.request_cache.initialize(rank=0)
        try:
            replay = await resumed_runner._cached_request(
                payload.copy(), asyncio.Semaphore(1), server, doc_id="doc-1", rollout_idx=0, chunk_index=0
            )
            assert replay.model == "test-model"
            assert server.make_request.await_count == 1
        finally:
            await resumed_runner.request_cache.close()

    asyncio.run(run_test())


def test_saved_rollout_result_includes_model(tmp_path: Path) -> None:
    """Returning InferenceResult directly writes model alongside the generated text."""
    runner = make_runner(tmp_path)
    documents = [Document(text="hello", id="doc-1")]
    asyncio.run(runner.run_async(documents, rank=0))
    output_files = list((tmp_path / "output").glob("*.jsonl"))
    assert len(output_files) == 1
    saved = json.loads(output_files[0].read_text())
    assert saved["metadata"]["rollout_results"][0]["model"] == "test-model"
