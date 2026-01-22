import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from datatrove.pipeline.inference.distributed.utils import (
    get_master_node_host,
    get_node_hosts,
    get_node_rank,
)
from datatrove.pipeline.inference.servers import InferenceServer
from datatrove.utils._import_utils import check_required_dependencies


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig


class SGLangServer(InferenceServer):
    """SGLang inference server implementation."""

    def __init__(self, config: "InferenceConfig", rank: int):
        """
        Initialize SGLang server.

        Args:
            config: InferenceConfig containing all server configuration parameters
            rank: Rank of the server instance
        """
        # Check required dependencies for SGLang server
        check_required_dependencies("SGLang server", ["sglang"])
        super().__init__(config, rank)

    async def start_server(self):
        n_nodes = len(get_node_hosts())
        if n_nodes <= 1:
            return await self.create_sglang_task()

        # Multi-node setup: configure distributed parameters
        dist_init_addr = get_master_node_host()
        node_rank = get_node_rank()

        return await self.create_sglang_task(n_nodes, node_rank, dist_init_addr)

    async def create_sglang_task(self, n_nodes: int = 1, node_rank: int = 0, dist_init_addr: str = "localhost"):
        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.config.model_name_or_path,
            "--allow-auto-truncate",
            "--context-length",
            str(self.config.model_max_context),
            "--port",
            str(self._port),
            "--log-level-http",
            "warning",
            "--nnodes",
            str(n_nodes),
            "--node-rank",
            str(node_rank),
            "--dist-init-addr",
            f"{dist_init_addr}:{self.config.master_port}",
        ]

        model_kwargs = self.config.model_kwargs.copy() if self.config.model_kwargs else {}
        # parallelism settings
        if self.config.tp > 1 and "tp-size" not in model_kwargs:
            model_kwargs["tp-size"] = self.config.tp
        if self.config.dp > 1 and "dp-size" not in model_kwargs:
            model_kwargs["dp-size"] = self.config.dp
        if self.config.pp > 1 and "pp-size" not in model_kwargs:
            model_kwargs["pp-size"] = self.config.pp

        # set kwargs
        if model_kwargs:
            for k, v in model_kwargs.items():
                if v is True:
                    cmd.append(f"--{k}")  # Boolean flag: e.g., --enforce-eager
                elif v is False:
                    cmd.append(f"--no-{k}")  # Negated flag: e.g., --no-enforce-eager
                else:
                    cmd.append(f"--{k}={v}")

        server_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return server_process

    async def monitor_health(self):
        """Monitor the health of the SGLang server, exiting or raising an exception if the server is not healthy"""
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

            if "The server is fired up and ready to roll!" in line:
                server_printed_ready_message = True
                logger.info("SGLang server startup complete")

            # Check for common SGLang errors
            if "Detected errors during sampling" in line:
                logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
                raise RuntimeError("Sampling errors detected, model is probably corrupt")
            elif "IndexError: list index out of range" in line:
                raise RuntimeError("IndexError in model, model is probably corrupt")

            # TODO: We should handle the torch.distributed fail, so that we can restart the server.

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

        if self._server_process is not None:
            stdout_task = asyncio.create_task(read_stream(self._server_process.stdout))
            stderr_task = asyncio.create_task(read_stream(self._server_process.stderr))
            try:
                await asyncio.gather(stdout_task, stderr_task, self._server_process.wait())
            finally:
                stdout_task.cancel()
                stderr_task.cancel()
                await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task, return_exceptions=True), timeout=5.0)
