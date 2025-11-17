import asyncio
import atexit
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

from loguru import logger

from datatrove.pipeline.inference.servers import InferenceServer
from datatrove.pipeline.inference.distributed.utils import (
    get_distributed_environment,
    get_master_node_ip,
    get_number_of_nodes,
    is_master_node,
)
from datatrove.pipeline.inference.distributed.ray import (
    init_ray_master,
    init_ray_worker,
    monitor_ray_cluster_health,
    cleanup_ray,
)
from datatrove.utils._import_utils import check_required_dependencies


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig

class VLLMServer(InferenceServer):
    """VLLM inference server implementation."""

    def __init__(self, config: "InferenceConfig"):
        """
        Initialize VLLM server.

        Args:
            config: InferenceConfig containing all server configuration parameters
        """
        # Check required dependencies for VLLM server
        check_required_dependencies("VLLM server", ["vllm"])
        super().__init__(config)

    @contextmanager
    def init_distributed_context(self):
        """Initialize distributed environment for VLLM.
        
        If in SLURM environment:
        - Master node: Initializes Ray cluster
        - Worker nodes: Connects to Ray cluster in subprocess
        """
        
        n_nodes = get_number_of_nodes()
        env = get_distributed_environment()
        if env != "SLURM" or n_nodes <= 1:
            # Only initialize Ray in SLURM environment with multiple nodes
            yield; return
        
        master_ip = get_master_node_ip()
        is_master = is_master_node()
        expected_workers = n_nodes  # We spawn worker on the master as well
        master_port = self.config.master_port
        timeout = self.config.distributed_init_timeout
        
        try:
            if is_master:
                init_ray_master(
                    master_port=master_port,
                    timeout=timeout,
                    expected_workers=expected_workers,
                )
            else:
                init_ray_worker(
                    master_ip=master_ip,
                    master_port=master_port,
                    init_timeout=timeout,
                )
                monitor_ray_cluster_health(check_interval=timeout)
            yield
        finally:
            cleanup_ray()
            

    async def start_server_task(self) -> None:
        """Start the VLLM server process."""

        cmd = [
            "vllm",
            "serve",
            self.config.model_name_or_path,
            "--port",
            str(self.port),
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
        self.server_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Ensure the subprocess is terminated on exit
        def _kill_proc():
            if self.server_process:
                self.server_process.terminate()

        atexit.register(_kill_proc)

        server_printed_ready_message = False

        # Create dedicated logger for server output
        server_logger = self._create_server_logger(getattr(self, "rank", 0))

        async def process_line(line):
            nonlocal server_printed_ready_message

            # Always log to file if server logger is available
            if server_logger:
                server_logger.info(line)

            # if the server hasn't initialized yet, log all the lines to the main logger also
            if not server_printed_ready_message:
                logger.info(line)

            if "Application startup complete" in line or "Uvicorn running on" in line:
                server_printed_ready_message = True
                logger.info("VLLM server startup complete")

            # Check for common VLLM errors
            if "CUDA out of memory" in line:
                logger.error("CUDA out of memory error detected")
                if self.server_process:
                    self.server_process.terminate()
            elif "RuntimeError" in line and "CUDA" in line:
                logger.error("CUDA runtime error detected")
                if self.server_process:
                    self.server_process.terminate()

        async def read_stream(stream):
            while True:
                line = await stream.readline()
                if not line:
                    break
                try:
                    line = line.decode("utf-8").rstrip()
                    await process_line(line)
                except Exception as ex:
                    logger.warning(f"Got {ex} when reading log line from inference server, skipping")

        # Start tasks to read stdout and stderr
        stdout_task = asyncio.create_task(read_stream(self.server_process.stdout))
        stderr_task = asyncio.create_task(read_stream(self.server_process.stderr))

        try:
            await self.server_process.wait()
        except asyncio.CancelledError:
            if self.server_process:
                self.server_process.terminate()
            raise

        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
