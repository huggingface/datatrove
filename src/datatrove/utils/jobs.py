import json
import os
import subprocess


def is_local_job_complete(logging_dir):
    if not os.path.exists(os.path.join(logging_dir, "completions")):
        return False
    with open(os.path.join(logging_dir, "executor.json")) as f:
        executor_data = json.load(f)
    return len(os.listdir(os.path.join(logging_dir, "completions"))) == executor_data["world_size"]


def get_running_slurm_jobs(partition=None):
    command = ["squeue", "--me", "--format=%j", "--noheader"]
    if partition:
        command.append(f"--partition={partition}")
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    job_names = result.stdout.splitlines()
    return job_names


def get_num_slurm_jobs(partition=None):
    command = ["squeue", "-u", os.environ["USER"], "-t", "pending,running", "--array", "--format=%.10t"]
    if partition:
        command.extend(["-p", partition])
    output = subprocess.check_output(command)
    num_jobs = int(output.strip())
    return num_jobs
