import asyncio
import os
from typing import TYPE_CHECKING

from loguru import logger

from datatrove.pipeline.inference.distributed.ray import (
    cleanup_ray,
    init_ray_master,
    init_ray_worker,
    monitor_ray_cluster_health,
    monitor_ray_workers,
)
from datatrove.pipeline.inference.distributed.utils import (
    get_distributed_environment,
    get_master_node_host,
    get_node_hosts,
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
            for k, v in model_kwargs.items():
                if v is True:
                    cmd.append(f"--{k}")  # Boolean flag: e.g., --enforce-eager
                elif v is False:
                    cmd.append(f"--no-{k}")  # Negated flag: e.g., --no-enforce-eager
                else:
                    cmd.append(f"--{k}={v}")

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

    async def start_server(self) -> asyncio.subprocess.Process | None:
        """Start the VLLM server process, handling distributed setup if needed."""
        n_nodes = len(get_node_hosts())
        if n_nodes <= 1:
            return await self._start_vllm_task()

        # VLLM has ray executor, which will create placement groups itself, however
        # in ray executor manages placement groups itself and expects to run the workers
        # itself. This is not compatible with VLLM multi-node setup as again VLLM expects to manage
        # workers placement groups itself.
        if n_nodes > 1 and get_distributed_environment() == "RAY":
            raise RuntimeError(
                "Datatrove Ray distributed executor doesn't support multi-node setup by specifying nodes=2+. Please unset the nodes=2+ parameter and let VLLM to allocate the number of nodes itself."
            )

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
            elif (
                "required The number of required GPUs exceeds the total number of available GPUs in the placement"
                in line
            ):
                raise RuntimeError("Not enough GPUs available for the placement")

            # TODO: We should handle the case when the ray fails, so that we can try to restart the server.

        async def read_stream(stream):
            while True:
                line = await stream.readline()
                if not line:
                    break
                line = line.decode("utf-8").rstrip()
                await process_line(line)

        async def monitor_ray_workers_after_server_ready():
            await self._server_ready
            await monitor_ray_workers(expected_workers=len(get_node_hosts()))

        if self._server_process is not None:
            tasks = [
                asyncio.create_task(read_stream(self._server_process.stdout)),
                asyncio.create_task(read_stream(self._server_process.stderr)),
                asyncio.create_task(self._server_process.wait()),
            ]

            # Only add ray monitoring task in distributed setting (nodes > 1)
            if len(get_node_hosts()) > 1:
                monitor_ray_task = asyncio.create_task(monitor_ray_workers_after_server_ready())
                tasks.append(monitor_ray_task)

            try:
                # Any exception raising or task completion from the set should cause the monitor to fail, we thus wait for first occurance of it.
                # Note this will not raise the exception
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # This will raise the exception if any of the tasks failed
                if done:
                    for task in done:
                        task.result()
            finally:
                # No need to deal with server_process itself as we do kill it in server_cleanup
                for task in tasks:
                    task.cancel()
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)

    async def server_cleanup(self):
        await super().server_cleanup()
        if len(get_node_hosts()) > 1:
            cleanup_ray()
