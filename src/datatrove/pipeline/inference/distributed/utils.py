import os
import socket
import subprocess
from typing import Literal


def _expand_slurm_nodelist(nodelist: str) -> list[str]:
    """Expand SLURM nodelist (which may contain range notation) to list of hostnames.
    
    Uses `scontrol show hostnames` to properly expand SLURM nodelist format like
    'ip-26-0-164-[45-46]' into individual hostnames.
    """
    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        capture_output=True,
        text=True,
        check=True,
    )
    hostnames = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    return hostnames


def _resolve_hostname_to_ip(hostname: str) -> str:
    """Resolve a hostname to its IP address."""
    try:
        # Try getent hosts first (more reliable in some environments)
        result = subprocess.run(
            ["getent", "hosts", hostname],
            capture_output=True,
            text=True,
            check=True,
        )
        ip = result.stdout.split()[0]
        return ip
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to socket.gethostbyname
        try:
            ip = socket.gethostbyname(hostname)
            return ip
        except socket.gaierror:
            # If resolution fails, return hostname as-is (might work for some setups)
            return hostname


def get_master_node_ip() -> str:
    """Return IP address of the master node."""
    if get_distributed_environment() == "SLURM":
        nodelist = os.environ["SLURM_NODELIST"]
        hostnames = _expand_slurm_nodelist(nodelist)
        if not hostnames:
            raise RuntimeError(f"Failed to expand SLURM nodelist: {nodelist}")
        master_hostname = hostnames[0]
        return _resolve_hostname_to_ip(master_hostname)
    elif get_distributed_environment() == "RAY":
        return os.environ["RAY_NODELIST"].split(",")[0]
    else:
        return "localhost"


def get_available_cpus_per_node() -> int:
    """Return the number of available CPUs in the distributed environment."""
    if get_distributed_environment() == "SLURM":
        return int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    raise NotImplementedError("Only SLURM distributed environment is supported yet.")

def get_available_memory_per_node() -> int:
    """Return the amount of available memory (MB) in the distributed environment."""
    cpus = get_available_cpus_per_node()
    mem_per_cpu = int(os.environ.get("SLURM_MEM_PER_CPU", 0))
    return cpus * mem_per_cpu

def get_available_gpus_per_node() -> int:
    """Return the number of available GPUs in the distributed environment."""
    if get_distributed_environment() == "SLURM":
        return int(os.environ.get("SLURM_GPUS_ON_NODE", 0))
    raise NotImplementedError("Only SLURM distributed environment is supported yet.")

def is_master_node() -> bool:
    """Return True if the current node is the master node, False otherwise."""
    if get_distributed_environment() == "SLURM":
        return int(os.environ.get("SLURM_NODEID", 0)) == 0
    elif get_distributed_environment() == "RAY":
        return int(os.environ.get("RAY_NODEID", 0)) == 0
    else:
        return True

def get_number_of_nodes() -> int:
    """Return the number of nodes in the distributed environment."""
    if get_distributed_environment() == "SLURM":
        return int(os.environ.get("SLURM_JOB_NUM_NODES", 0))

    raise NotImplementedError("Only SLURM distributed environment is supported yet.")

def get_node_ips() -> list[str]:
    """Return list of IP addresses of the nodes in the distributed environment."""
    if get_distributed_environment() == "SLURM":
        nodelist = os.environ["SLURM_NODELIST"]
        hostnames = _expand_slurm_nodelist(nodelist)
        return [_resolve_hostname_to_ip(hostname) for hostname in hostnames]
    elif get_distributed_environment() == "RAY":
        return os.environ["RAY_NODELIST"].split(",")
    else:
        return ["localhost"]

def get_distributed_environment() -> Literal["SLURM", "RAY"] | None:
    """Return type of the distributed environment, this means either "SLURM", "RAY", or None.
    """
    if "SLURM_NODEID" in os.environ:
        return "SLURM"
    elif "RAY_NODEID" in os.environ:
        return "RAY"
    else:
        return None

