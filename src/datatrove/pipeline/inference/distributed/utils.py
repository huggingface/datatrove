import os
from typing import Literal


def get_available_cpus_per_node() -> int:
    """Return the number of available CPUs in the distributed environment.
    -1 signifies unknown CPUs.
    """

    return int(os.environ.get("DATATROVE_CPUS_PER_TASK", -1))


def get_available_memory_per_node() -> int:
    """Return the amount of available memory (MB) in the distributed environment.
    -1 signifies unknown memory.
    """
    mem_per_cpu = int(os.environ.get("DATATROVE_MEM_PER_CPU", -1))
    if mem_per_cpu == -1:
        return -1

    cpus = get_available_cpus_per_node()
    return cpus * mem_per_cpu


def get_available_gpus_per_node() -> int:
    """Return the number of available GPUs in the distributed environment.
    -1 signifies unknown GPUs.
    """
    return int(os.environ.get("DATATROVE_GPUS_ON_NODE", -1))


def get_node_rank() -> int:
    """Return the rank of the current node (0 for master, 1, 2, etc. for workers). -1 stands for single-node mode."""
    return int(os.environ.get("DATATROVE_NODE_RANK", -1))


def is_master_node() -> bool:
    """Return True if the current node is the master node, False otherwise."""
    return get_node_rank() <= 0


def get_node_hosts() -> list[str]:
    """Return list of hosts of the nodes in the distributed environment."""
    return os.environ.get("DATATROVE_NODE_IPS", "").split(",")


def get_distributed_environment() -> Literal["SLURM", "RAY", "LOCAL"]:
    """Return type of the distributed environment, this means either "SLURM", "RAY", or "LOCAL"."""
    return os.environ.get("DATATROVE_EXECUTOR", "LOCAL")


def get_master_node_host() -> str:
    """Return the hostname of the master node."""
    return get_node_hosts()[0]
