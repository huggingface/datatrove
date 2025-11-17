"""Ray cluster initialization and management functions."""

import subprocess
import time
from loguru import logger

from datatrove.pipeline.inference.distributed.utils import (
    get_available_cpus_per_node,
    get_available_gpus_per_node,
    get_available_memory_per_node,
    get_master_node_ip,
    get_number_of_nodes,
    is_master_node,
)
from datatrove.utils._import_utils import check_required_dependencies

# Constants
WORKER_CHECK_INTERVAL_SECONDS = 2.0


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


def _count_alive_worker_nodes(master_ip: str) -> int:
    """Count the number of alive worker nodes in the Ray cluster.
    
    Args:
        master_ip: IP address of the master node (to exclude from count)
    
    Returns:
        Number of alive worker nodes
    """
    import ray
    nodes = ray.nodes()
    alive_worker_nodes = [
        node for node in nodes 
        if node["Alive"] and node["NodeManagerAddress"] != master_ip
    ]
    return len(alive_worker_nodes)


def _start_ray_head_node(master_port: int, object_store_memory: int, timeout: float) -> None:
    """Start the Ray head node using CLI.
    
    Args:
        master_port: Port for the Ray cluster
        object_store_memory: Object store memory in bytes
        timeout: Timeout in seconds for starting Ray
    
    Raises:
        RuntimeError: If Ray head node fails to start
        subprocess.TimeoutExpired: If the start command times out
    """
    ray_start_cmd = _build_ray_head_start_command(master_port, object_store_memory)
    
    logger.info(f"Running: {' '.join(ray_start_cmd)}")
    result = subprocess.run(
        ray_start_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to start Ray head node. Return code: {result.returncode}, "
            f"stderr: {result.stderr}, stdout: {result.stdout}"
        )
    
    logger.info("Ray head node started successfully")


def _wait_for_workers(expected_workers: int, master_ip: str, timeout: float) -> None:
    """Wait for worker nodes to join the Ray cluster.
    
    Args:
        expected_workers: Expected number of worker nodes
        master_ip: IP address of the master node
        timeout: Maximum time to wait in seconds
    
    Raises:
        RuntimeError: If workers don't join within the timeout period
    """
    import ray
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            connected_workers = _count_alive_worker_nodes(master_ip)
            
            if connected_workers >= expected_workers:
                nodes = ray.nodes()
                total_alive_nodes = len([n for n in nodes if n['Alive']])
                logger.info(
                    f"All {expected_workers} worker nodes are online. "
                    f"Total nodes in cluster: {total_alive_nodes}"
                )
                logger.info(ray.nodes())
                return
            
            elapsed_time = int(time.time() - start_time)
            logger.info(
                f"Waiting for workers... {connected_workers}/{expected_workers} connected. "
                f"Elapsed: {elapsed_time}s"
            )
            time.sleep(WORKER_CHECK_INTERVAL_SECONDS)
        except Exception as e:
            logger.warning(f"Error checking Ray cluster state: {e}")
            time.sleep(WORKER_CHECK_INTERVAL_SECONDS)
    
    # Timeout reached
    connected_workers = _count_alive_worker_nodes(master_ip)
    raise RuntimeError(
        f"Timeout ({timeout}s) waiting for workers to join Ray cluster. "
        f"Only {connected_workers}/{expected_workers} workers connected."
    )


def init_ray_master(
    master_port: int,
    timeout: float,
    expected_workers: int | None = None,
) -> None:
    """Initialize Ray cluster on the master node.
    
    Starts Ray head node using CLI and waits for worker nodes to join.
    
    Args:
        master_port: Port for the Ray cluster
        timeout: Timeout in seconds for starting Ray and waiting for workers
        expected_workers: Expected number of worker nodes. If None, uses number of nodes.
    
    Raises:
        RuntimeError: If Ray head node fails to start or workers don't join in time
    """
    check_required_dependencies("Ray", ["ray"])
    import ray
    
    master_ip = get_master_node_ip()
    expected_workers = expected_workers or get_number_of_nodes()
    object_store_memory = calculate_object_store_memory()
    
    logger.info(f"Starting Ray head node on master (IP: {master_ip}, port: {master_port})")
    logger.info(f"Expecting {expected_workers} worker nodes to join")
    
    try:
        _start_ray_head_node(master_port, object_store_memory, timeout)
        
        # Connect to the Ray cluster
        ray.init(address="auto", ignore_reinit_error=True)
        logger.info("Connected to Ray cluster")
        
        # Wait for workers to come online
        _wait_for_workers(expected_workers, master_ip, timeout)
        
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Timeout ({timeout}s) starting Ray head node") from e
    except Exception as e:
        logger.error(f"Failed to initialize Ray cluster: {e}")
        raise


def init_ray_worker(
    master_ip: str,
    master_port: int,
    init_timeout: float,
) -> None:
    """Initialize Ray worker node and connect to master.
    
    Starts Ray worker node using CLI and connects to the master Ray cluster.
    After connecting, waits for init_timeout seconds to allow cluster initialization.
    
    Args:
        master_ip: IP address of the master node
        master_port: Port of the Ray cluster
        init_timeout: Timeout in seconds for connecting to Ray cluster and initialization wait
    
    Raises:
        RuntimeError: If Ray worker node fails to start
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
    
    logger.info(f"Running: {' '.join(ray_start_cmd)}")
    result = subprocess.run(
        ray_start_cmd,
        capture_output=True,
        text=True,
        timeout=init_timeout,
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to start Ray worker node. Return code: {result.returncode}, "
            f"stderr: {result.stderr}, stdout: {result.stdout}"
        )
    
    logger.info(f"Worker node successfully connected to Ray cluster at {master_ip}:{master_port}")

def monitor_ray_cluster_health(check_interval: float = 5.0) -> None:
    """Monitor Ray cluster health by running ray health-check command.
    
    This function runs in a loop, checking every check_interval seconds if the Ray cluster
    is still healthy by calling `ray health-check --skip-version-check`. If the health check
    fails, it returns to signal the process should exit.
    
    Args:
        check_interval: Interval in seconds between health checks (default: 5.0)
    """
    logger.info(f"Starting Ray cluster health monitoring (checking every {check_interval}s)")
    
    while True:
        try:
            # Run ray health-check command
            result = subprocess.run(
                ["ray", "health-check", "--skip-version-check"],
                capture_output=True,
                text=True,
                timeout=check_interval,
            )
            
            if result.returncode != 0:
                # Health check failed, cluster is dead
                logger.warning(
                    f"Ray cluster health check failed (return code: {result.returncode}). "
                    f"stderr: {result.stderr}, stdout: {result.stdout}. Exiting health monitor."
                )
                return
            
            time.sleep(check_interval)
            
        except subprocess.TimeoutExpired:
            # Health check timed out, treat as failure
            logger.warning(f"Ray cluster health check timed out. Exiting health monitor.")
            return
        except Exception as e:
            # Any other exception means health check failed
            logger.warning(f"Ray cluster health check failed: {e}. Exiting health monitor.")
            return


def cleanup_ray() -> None:
    """Cleanup Ray cluster by stopping Ray on the current node.
    
    Stops the Ray runtime on the current node using `ray stop`.
    """
    logger.info("Cleaning up Ray cluster")
    try:
        subprocess.run(["ray", "stop"], timeout=30)
        logger.info("Ray cluster stopped")
    except Exception as e:
        logger.warning(f"Error stopping Ray cluster: {e}")

