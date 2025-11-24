import os
import subprocess
from typing import Literal





def get_available_cpus_per_node() -> int:
    """Return the number of available CPUs in the distributed environment.
    -1 signifies unknown CPUs.
    """

    return int(os.environ["DATATROVE_CPUS_PER_TASK"])

def get_available_memory_per_node() -> int:
    """Return the amount of available memory (MB) in the distributed environment.
    -1 signifies unknown memory.
    """
    mem_per_cpu = int(os.environ["DATATROVE_MEM_PER_CPU"])
    if mem_per_cpu == -1:
        return -1

    cpus = get_available_cpus_per_node()
    return cpus * mem_per_cpu

def get_available_gpus_per_node() -> int:
    """Return the number of available GPUs in the distributed environment.
    -1 signifies unknown GPUs.
    """
    return int(os.environ["DATATROVE_GPUS_ON_NODE"])


def get_node_rank() -> int:
    """Return the rank of the current node (0 for master, 1, 2, etc. for workers). -1 stands for single-node mode."""
    return int(os.environ["DATATROVE_NODE_RANK"])


def is_master_node() -> bool:
    """Return True if the current node is the master node, False otherwise."""
    return get_node_rank() <= 0


def get_node_hosts() -> list[str]:
    """Return list of hosts of the nodes in the distributed environment."""
    return os.environ["DATATROVE_NODE_IPS"].split(",")


def get_distributed_environment() -> Literal["SLURM", "RAY", "LOCAL"]:
    """Return type of the distributed environment, this means either "SLURM", "RAY", or "LOCAL"."""
    return os.environ["DATATROVE_EXECUTOR"]


def get_master_node_host() -> str:
    """Return the hostname of the master node."""
    return get_node_hosts()[0]