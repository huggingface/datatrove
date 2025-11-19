import os
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


def get_master_node_host() -> str:
    """Return hostname of the master node."""
    if get_distributed_environment() == "SLURM":
        nodelist = os.environ["SLURM_NODELIST"]
        hostnames = _expand_slurm_nodelist(nodelist)
        if not hostnames:
            raise RuntimeError(f"Failed to expand SLURM nodelist: {nodelist}")
        master_hostname = hostnames[0]
        return master_hostname
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


def get_node_rank() -> int:
    """Return the rank of the current node (0 for master, 1, 2, etc. for workers)."""
    if get_distributed_environment() == "SLURM":
        return int(os.environ.get("SLURM_NODEID", 0))
    elif get_distributed_environment() == "RAY":
        return int(os.environ.get("RAY_NODEID", 0))
    else:
        return 0


def is_master_node() -> bool:
    """Return True if the current node is the master node, False otherwise."""
    return get_node_rank() == 0


def get_number_of_nodes() -> int:
    """Return the number of nodes in the distributed environment."""
    if get_distributed_environment() == "SLURM":
        return int(os.environ.get("SLURM_JOB_NUM_NODES", 0))

    return len(get_node_hosts())


def get_node_hosts() -> list[str]:
    """Return list of hosts of the nodes in the distributed environment."""
    if get_distributed_environment() == "SLURM":
        nodelist = os.environ["SLURM_NODELIST"]
        hostnames = _expand_slurm_nodelist(nodelist)
        return hostnames
    elif get_distributed_environment() == "RAY":
        return os.environ["RAY_NODELIST"].split(",")
    else:
        return ["localhost"]


def get_job_id() -> str:
    """Return the job ID of the distributed environment."""
    if get_distributed_environment() == "SLURM":
        return os.environ.get("SLURM_JOB_ID", "unknown")
    elif get_distributed_environment() == "RAY":
        return os.environ.get("RAY_JOB_ID", "unknown")
    else:
        return "unknown"


def get_distributed_environment() -> Literal["SLURM", "RAY"] | None:
    """Return type of the distributed environment, this means either "SLURM", "RAY", or None."""
    if "SLURM_NODEID" in os.environ:
        return "SLURM"
    elif "RAY_NODEID" in os.environ:
        return "RAY"
    else:
        return None
