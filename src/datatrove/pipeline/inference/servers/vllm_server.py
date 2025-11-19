import asyncio
import os
from typing import TYPE_CHECKING

from loguru import logger

from datatrove.pipeline.inference.distributed.ray import (
    cleanup_ray,
    init_ray_master,
    init_ray_worker,
    monitor_ray_cluster_health,
)
from datatrove.pipeline.inference.distributed.utils import (
    get_master_node_host,
    get_number_of_nodes,
    is_master_node,
)
from datatrove.pipeline.inference.servers import InferenceServer
from datatrove.utils._import_utils import check_required_dependencies


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig


class VLLMServer(InferenceServer):
    """VLLM inference server implementation."""

    def __init__(self, config: "InferenceConfig", rank: int):
        """
        Initialize VLLM server.

        Args:
            config: InferenceConfig containing all server configuration parameters
        """
        # Check required dependencies for VLLM server
        check_required_dependencies("VLLM server", ["vllm"])
        super().__init__(config, rank)

    async def _start_vllm_task(self) -> asyncio.subprocess.Process:
        """Start the VLLM server process."""
        cmd = [
            "vllm",
            "serve",
            self.config.model_name_or_path,
            "--port",
            str(self._port),
            "--max-model-len",
            str(self.config.model_max_context),
            "--trust-remote-code",
            "--disable-log-requests",  # Disable verbose request logging
            "--disable-uvicorn-access-log",
        ]

        model_kwargs = self.config.model_kwargs.copy() if self.config.model_kwargs else {}
        # parallelism settings
        if self.config.tp > 1 and "tensor-parallel-size" not in model_kwargs:
            model_kwargs["tensor-parallel-size"] = self.config.tp
        if self.config.dp > 1 and "data-parallel-size" not in model_kwargs:
            model_kwargs["data-parallel-size"] = self.config.dp
        if self.config.pp > 1 and "pipeline-parallel-size" not in model_kwargs:
            model_kwargs["pipeline-parallel-size"] = self.config.pp
        # set kwargs
        if model_kwargs:
            cmd.extend([f"--{k}={v}" for k, v in model_kwargs.items()])

        logger.debug(f"Starting VLLM server with command: {' '.join(cmd)}")
        env = os.environ.copy()
        # transformers pulls in TensorFlow by default, which adds tens of seconds of startup time
        # (we measured ~70-80s at tp=2 on H100). These env vars keep it in PyTorch-only mode so
        # vLLM initializes much faster without affecting throughput.
        env.setdefault("USE_TF", "0")
        env.setdefault("TRANSFORMERS_NO_TF", "1")
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def start_server_task(self) -> asyncio.subprocess.Process | None:
        """Start the VLLM server process, handling distributed setup if needed."""
        n_nodes = get_number_of_nodes()
        if n_nodes <= 1:
            return await self._start_vllm_task()

        master_ip = get_master_node_host()
        is_master = is_master_node()
        expected_workers = n_nodes  # We spawn worker on the master as well
        master_port = self.config.master_port

        if is_master:
            await init_ray_master(
                master_port=master_port,
                expected_workers=expected_workers,
            )
            return await self._start_vllm_task()
        else:
            await init_ray_worker(
                master_ip=master_ip,
                master_port=master_port,
            )
            return None  # Worker nodes don't start a server process

    async def monitor_health(self):
        """Monitor the health of the VLLM server, exiting or raising an exception if the server is not healthy"""
        if not self._is_master:
            # This will block until ray cluster fails
            await monitor_ray_cluster_health()
            return

        server_printed_ready_message = False

        async def process_line(line):
            nonlocal server_printed_ready_message

            # Always log to file if server logger is available
            logger_instance = self.server_logger()
            if logger_instance:
                logger_instance.info(line)

            # if the server hasn't initialized yet, log all the lines to the main logger also
            if not server_printed_ready_message:
                logger.info(line)

            if "Application startup complete" in line or "Uvicorn running on" in line:
                server_printed_ready_message = True
                logger.info("VLLM server startup complete")

            # Check for common VLLM errors
            if "CUDA out of memory" in line:
                raise RuntimeError("CUDA out of memory error detected")
            elif "RuntimeError" in line and "CUDA" in line:
                raise RuntimeError("CUDA runtime error detected")

            # Not enough gpus for TP/DP/PP
            elif "required The number of required GPUs exceeds the total number of available GPUs in the placemen":
                raise RuntimeError("Not enough GPUs available for the placement")

        async def read_stream(stream):
            while True:
                line = await stream.readline()
                if not line:
                    break
                line = line.decode("utf-8").rstrip()
                await process_line(line)

        if self._server_process is not None:
            stdout_task = asyncio.create_task(read_stream(self._server_process.stdout))
            stderr_task = asyncio.create_task(read_stream(self._server_process.stderr))
            try:
                # Here he explicity want to the process to raise an exception if it terminates unexpectedly
                await asyncio.gather(stdout_task, stderr_task, self._server_process.wait())
            finally:
                # No need to deal with server_process itself as we do kill it in server_cleanup
                stdout_task.cancel()
                stderr_task.cancel()
                await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task, return_exceptions=True), timeout=5.0)

    async def server_cleanup(self):
        await super().server_cleanup()
        if self._is_master:
            cleanup_ray()
