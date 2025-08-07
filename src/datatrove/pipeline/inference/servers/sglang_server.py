import asyncio
import atexit

from loguru import logger

from datatrove.pipeline.inference.servers import InferenceServer


class SGLangServer(InferenceServer):
    """SGLang inference server implementation."""

    async def start_server_task(self) -> None:
        """Start the SGLang server process."""

        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_name_or_path,
            "--allow-auto-truncate",
            "--context-length", str(self.max_context),
            "--port",
            str(self.port),
            "--log-level-http",
            "warning",
        ]
        if self.model_kwargs:
            cmd.extend([f"--{k}={v}" for k, v in self.model_kwargs.items()])

        self.server_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Ensure the subprocess is terminated on exit
        def _kill_proc():
            if self.server_process and self.server_process.returncode is None:
                self.server_process.terminate()

        atexit.register(_kill_proc)

        server_printed_ready_message = False

        # Create dedicated logger for server output
        server_logger = self._create_server_logger(getattr(self, 'rank', 0))

        def process_line(line):
            nonlocal server_printed_ready_message

            # Always log to file if server logger is available
            if server_logger:
                server_logger.info(line)

            # if the server hasn't initialized yet, log all the lines to the main logger also
            if not server_printed_ready_message:
                logger.info(line)

            if "The server is fired up and ready to roll!" in line:
                server_printed_ready_message = True
                logger.info("SGLang server startup complete")

            # Check for common SGLang errors
            if "Detected errors during sampling" in line:
                logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
                if self.server_process:
                    self.server_process.terminate()
            elif "IndexError: list index out of range" in line:
                logger.error("IndexError in model, restarting server")
                if self.server_process:
                    self.server_process.terminate()

        async def read_stream(stream):
            while True:
                line = await stream.readline()
                if not line:
                    break
                try:
                    line = line.decode("utf-8").rstrip()
                    process_line(line)
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
