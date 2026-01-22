import asyncio
import sys
from typing import TYPE_CHECKING

from loguru import logger

from datatrove.pipeline.inference.servers import InferenceServer


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig


class CustomServer(InferenceServer):
    """Custom inference server implementation."""

    def __init__(self, config: "InferenceConfig", rank: int):
        """
        Initialize Custom server.

        Args:
            config: InferenceConfig containing all server configuration parameters
            rank: Rank of the server
        """
        super().__init__(config, rank)
        if "server_script" not in self.config.model_kwargs:
            raise ValueError("server_script is not provided in the model_kwargs")
        self.server_script = self.config.model_kwargs["server_script"]
        del self.config.model_kwargs["server_script"]

    async def start_server(self) -> asyncio.subprocess.Process | None:
        """Start the Custom server process."""

        # Create the command to run the transformers_batching_app directly
        cmd = [
            sys.executable,
            self.server_script,
            "--port",
            str(self._port),
        ]

        # Add model kwargs if provided
        if self.config.model_kwargs:
            for k, v in self.config.model_kwargs.items():
                if v is True:
                    cmd.append(f"--{k}")  # Boolean flag: e.g., --enforce-eager
                elif v is False:
                    cmd.append(f"--no-{k}")  # Negated flag: e.g., --no-enforce-eager
                else:
                    cmd.append(f"--{k}={v}")

        logger.info(f"Starting Custom server with command: {' '.join(cmd)}")

        return await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def monitor_health(self):
        """Monitor the health of the Custom server, exiting or raising an exception if the server is not healthy"""
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
                logger.info("Custom server startup complete")

            # Check for common Transformers errors
            if "CUDA out of memory" in line:
                raise RuntimeError("CUDA out of memory error detected")
            elif "RuntimeError" in line and "CUDA" in line:
                raise RuntimeError("CUDA runtime error detected")
            elif "ImportError" in line or "ModuleNotFoundError" in line:
                # We raise RuntimeError so it's caught by the base class monitor loop
                raise RuntimeError("Import error detected - missing dependencies")

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
            tasks = [
                asyncio.create_task(read_stream(self._server_process.stdout)),
                asyncio.create_task(read_stream(self._server_process.stderr)),
                asyncio.create_task(self._server_process.wait()),
            ]

            try:
                # Any exception raising or task completion from the set should cause the monitor to fail
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if done:
                    for task in done:
                        task.result()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)  # type: ignore
