from unittest.mock import patch

from datatrove.utils.jobs import get_num_slurm_jobs


def test_get_num_slurm_jobs_counts_squeue_lines():
    with patch("datatrove.utils.jobs.subprocess.check_output", return_value="STATE\nR\nPD\n") as check_output:
        assert get_num_slurm_jobs() == 3

    check_output.assert_called_once_with(
        ["squeue", "--me", "-t", "pending,running", "--array", "--format=%.10t"],
        text=True,
    )


def test_get_num_slurm_jobs_passes_partition_as_single_arg():
    partition = "gpu; touch /tmp/datatrove-shell-injection"

    with patch("datatrove.utils.jobs.subprocess.check_output", return_value="STATE\n") as check_output:
        assert get_num_slurm_jobs(partition=partition) == 1

    check_output.assert_called_once_with(
        ["squeue", "-p", partition, "--me", "-t", "pending,running", "--array", "--format=%.10t"],
        text=True,
    )
