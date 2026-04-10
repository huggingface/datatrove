import unittest
from unittest.mock import AsyncMock, patch

from datatrove.pipeline.inference.run_inference import InferenceConfig
from datatrove.pipeline.inference.servers.vllm_server import VLLMServer


class TestVLLMServer(unittest.IsolatedAsyncioTestCase):
    async def test_start_vllm_task_uses_supported_cli_flags(self) -> None:
        config = InferenceConfig(server_type="vllm", model_name_or_path="Qwen/Qwen3.5-2B")
        server = VLLMServer(config, rank=0)
        server._port = 8123

        process = object()

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as create_subprocess_exec:
            result = await server._start_vllm_task()

        self.assertIs(result, process)
        args = create_subprocess_exec.await_args.args
        self.assertIn("--disable-uvicorn-access-log", args)
        self.assertNotIn("--disable-log-requests", args)
