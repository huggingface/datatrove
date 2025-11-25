"""Ray cluster initialization and management functions."""

import asyncio
import subprocess
import time

from loguru import logger

from datatrove.pipeline.inference.distributed.utils import (
    get_available_cpus_per_node,
    get_available_gpus_per_node,
    get_available_memory_per_node,
    get_master_node_host,
)
from datatrove.utils._import_utils import check_required_dependencies


# Constants
WORKER_CHECK_INTERVAL_SECONDS = 2.0
WORKER_CONNECTION_MAX_RETRIES = 5
WORKER_CONNECTION_RETRY_DELAY = 10.0
PROCESS_TIMEOUT = 10.0


def calculate_object_store_memory() -> int:
    """Calculate the object store memory for Ray.

    Calculates object store memory as the minimum of:
    - 30% of available memory
    - Size of /dev/shm (shared memory)
    - 200GB maximum cap

    Returns:
        Object store memory in bytes
    """
    available_memory_mb = get_available_memory_per_node()
    # 30% of available memory in bytes
    object_store_memory_30_percent = int(0.30 * available_memory_mb * 1024 * 1024)
    # Get shm size in bytes (fallback to a large value if /dev/shm doesn't exist)
    import shutil

    try:
        shm_size_bytes = shutil.disk_usage("/dev/shm").total
    except (OSError, FileNotFoundError):
        # If /dev/shm doesn't exist, use a large default (200G)
        shm_size_bytes = 200 * 1024 * 1024 * 1024
    # Cap at 200G (200 * 1024^3 bytes)
    max_object_store_memory = 200 * 1024 * 1024 * 1024
    # Take the minimum of all three constraints
    return min(object_store_memory_30_percent, shm_size_bytes, max_object_store_memory)


def _build_ray_head_start_command(master_port: int, object_store_memory: int) -> list[str]:
    """Build the Ray head node start command.

    Args:
        master_port: Port for the Ray cluster
        object_store_memory: Object store memory in bytes

    Returns:
        List of command arguments for subprocess
    """
    return [
        "ray",
        "start",
        "--head",
        "--port",
        str(master_port),
        "--disable-usage-stats",
        "--include-log-monitor=false",
        "--include-dashboard=false",
        "--num-cpus",
        str(get_available_cpus_per_node()),
        "--num-gpus",
        str(get_available_gpus_per_node()),
        "--object-store-memory",
        str(object_store_memory),
    ]


def _count_alive_worker_nodes() -> int:
    """Count the number of alive worker nodes in the Ray cluster.

    Args:
        master_ip: IP address of the master node (to exclude from count)

    Returns:
        Number of alive worker nodes
    """
    import ray

    nodes = ray.nodes()
    alive_worker_nodes = [node for node in nodes if node["Alive"]]
    return len(alive_worker_nodes)


async def _start_ray_head_node(master_port: int, object_store_memory: int) -> None:
    """Start the Ray head node using CLI.

    Args:
        master_port: Port for the Ray cluster
        object_store_memory: Object store memory in bytes

    Raises:
        RuntimeError: If Ray head node fails to start
        asyncio.TimeoutError: If the start command times out
    """
    ray_start_cmd = _build_ray_head_start_command(master_port, object_store_memory)

    logger.info(f"Running: {' '.join(ray_start_cmd)}")

    process = await asyncio.create_subprocess_exec(
        *ray_start_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=PROCESS_TIMEOUT)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(f"Timeout ({PROCESS_TIMEOUT}s) starting Ray head node")

    if process.returncode != 0:
        raise RuntimeError(
            f"Failed to start Ray head node. Return code: {process.returncode}, "
            f"stderr: {stderr.decode()}, stdout: {stdout.decode()}"
        )

    logger.info("Ray head node started successfully")


async def _wait_for_workers(expected_workers: int) -> None:
    """Wait for worker nodes to join the Ray cluster.

    Args:
        expected_workers: Expected number of worker nodes
        timeout: Maximum time to wait in seconds

    Raises:
        RuntimeError: If workers don't join within the timeout period
    """
    import ray

    start_time = time.time()
    max_time = (1 + WORKER_CONNECTION_MAX_RETRIES) * WORKER_CONNECTION_RETRY_DELAY
    connected_workers = _count_alive_worker_nodes()
    while time.time() - start_time < max_time:
        try:
            connected_workers = _count_alive_worker_nodes()

            if connected_workers >= expected_workers:
                logger.info(
                    f"All {expected_workers} worker nodes are online. Total nodes in cluster: {connected_workers}"
                )
                logger.info(ray.nodes())
                return

            elapsed_time = int(time.time() - start_time)
            logger.info(
                f"Waiting for workers... {connected_workers}/{expected_workers} connected. Elapsed: {elapsed_time}s"
            )
            await asyncio.sleep(WORKER_CHECK_INTERVAL_SECONDS)
        except Exception as e:
            logger.warning(f"Error checking Ray cluster state: {e}")
            await asyncio.sleep(WORKER_CHECK_INTERVAL_SECONDS)

    # Timeout reached
    raise RuntimeError(
        f"Timeout ({max_time}s) waiting for workers to join Ray cluster. "
        f"Only {connected_workers}/{expected_workers} workers connected."
    )


async def init_ray_master(master_port: int, expected_workers: int) -> None:
    """Initialize Ray cluster on the master node.

    Starts Ray head node using CLI and waits for worker nodes to join.

    Args:
        master_port: Port for the Ray cluster
        expected_workers: Expected number of worker nodes. If None, uses number of nodes.

    Raises:
        RuntimeError: If Ray head node fails to start or workers don't join in time
    """
    check_required_dependencies("Ray", ["ray"])
    import ray

    master_ip = get_master_node_host()
    object_store_memory = calculate_object_store_memory()

    logger.info(f"Starting Ray head node on master (IP: {master_ip}, port: {master_port})")
    logger.info(f"Expecting {expected_workers} worker nodes to join")

    try:
        await _start_ray_head_node(master_port, object_store_memory)

        # Connect to the Ray cluster
        ray.init(address="auto", ignore_reinit_error=True)
        # Wait for workers to come online
        await _wait_for_workers(expected_workers)

    except asyncio.TimeoutError as e:
        raise RuntimeError("Timeout (10.0s) starting Ray head node") from e
    except Exception as e:
        logger.error(f"Failed to initialize Ray cluster: {e}")
        raise


async def init_ray_worker(
    master_ip: str,
    master_port: int,
    max_retries: int = WORKER_CONNECTION_MAX_RETRIES,
    retry_delay: float = WORKER_CONNECTION_RETRY_DELAY,
) -> None:
    """Initialize Ray worker node and connect to master.
    Starts Ray worker node using CLI and connects to the master Ray cluster.
    Will retry up to max_retries times if connection fails.
    Args:
        master_ip: IP address of the master node
        master_port: Port of the Ray cluster
        max_retries: Maximum number of connection attempts (default: 5)
        retry_delay: Delay in seconds between retry attempts (default: 10.0)
    Raises:
        RuntimeError: If Ray worker node fails to start after all retries
    """
    object_store_memory = calculate_object_store_memory()

    logger.info(f"Worker node: Connecting to Ray cluster at {master_ip}:{master_port}")

    # Start Ray worker node using CLI
    ray_start_cmd = [
        "ray",
        "start",
        "--address",
        f"{master_ip}:{master_port}",
        "--disable-usage-stats",
        "--include-log-monitor=false",
        "--object-store-memory",
        str(object_store_memory),
        "--num-cpus",
        str(get_available_cpus_per_node()),
        "--num-gpus",
        str(get_available_gpus_per_node()),
    ]

    last_error = None

    for attempt in range(1, max_retries + 1):
        logger.info(f"Connection attempt {attempt}/{max_retries}: Running: {' '.join(ray_start_cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *ray_start_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=PROCESS_TIMEOUT)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                last_error = RuntimeError(f"Timeout ({PROCESS_TIMEOUT}s) connecting to Ray cluster")
                logger.warning(f"Attempt {attempt}/{max_retries} timed out. {last_error}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                continue

            if process.returncode != 0:
                last_error = RuntimeError(
                    f"Failed to start Ray worker node. Return code: {process.returncode}, "
                    f"stderr: {stderr.decode()}, stdout: {stdout.decode()}"
                )
                logger.warning(f"Attempt {attempt}/{max_retries} failed. {last_error}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                continue

            # Success!
            logger.info(f"Worker node successfully connected to Ray cluster at {master_ip}:{master_port}")
            return

        except Exception as e:
            last_error = RuntimeError(f"Failed to start Ray worker node: {e}")
            logger.warning(f"Attempt {attempt}/{max_retries} failed with exception. {last_error}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)

    # All retries exhausted
    raise RuntimeError(
        f"Failed to connect Ray worker to cluster after {max_retries} attempts. Last error: {last_error}"
    )


async def monitor_ray_cluster_health(check_interval: float = 30.0) -> None:
    """Monitor Ray cluster health by running ray health-check command.
    This function runs in a loop, checking every check_interval seconds if the Ray cluster
    is still healthy by calling `ray health-check --skip-version-check`. If the health check
    fails, it returns to signal the process should exit.
    Args:
        check_interval: Interval in seconds between health checks (default: 30.0)
    """
    logger.info(f"Starting Ray cluster health monitoring (checking every {check_interval}s)")

    while True:
        try:
            # Run ray health-check command
            process = await asyncio.create_subprocess_exec(
                "ray",
                "health-check",
                "--skip-version-check",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=5.0)
                logger.warning("Ray cluster health check timed out. Exiting health monitor.")
                return

            if process.returncode != 0:
                # Health check failed, cluster is dead
                logger.warning(
                    f"Ray cluster health check failed (return code: {process.returncode}). "
                    f"stderr: {stderr.decode()}, stdout: {stdout.decode()}. Exiting health monitor."
                )
                return

            await asyncio.sleep(check_interval)

        except Exception as e:
            # Any other exception means health check failed
            logger.warning(f"Ray cluster health check failed: {e}. Exiting health monitor.")
            return


async def monitor_ray_workers(expected_workers: int) -> None:
    """Monitor the number of ray nodes. Raises an exception if the number of nodes has dropped"""
    logger.info(f"Starting Ray nodes monitoring (expecting {expected_workers} workers)")

    while True:
        try:
            # Get cluster resources to check node count
            if _count_alive_worker_nodes() < expected_workers:
                raise RuntimeError("Number of ray nodes has dropped below expected")
            await asyncio.sleep(WORKER_CHECK_INTERVAL_SECONDS)

        except Exception as e:
            raise RuntimeError(f"Ray nodes monitoring failed: {e}") from e


def cleanup_ray() -> None:
    """Cleanup Ray cluster by stopping Ray on the current node.
    Stops the Ray runtime on the current node using `ray stop`.
    """
    import ray

    logger.info("Cleaning up Ray cluster")
    try:
        ray.shutdown()
        try:
            subprocess.run(["ray", "stop"], timeout=30, capture_output=True)
        except subprocess.TimeoutExpired:
            pass
        logger.info("Ray cluster stopped")
    except Exception:
        pass
