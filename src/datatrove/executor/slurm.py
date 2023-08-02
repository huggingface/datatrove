from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap

import dill
from loguru import logger

from datatrove.executor.base import PipelineExecutor


class SlurmPipelineExecutor(PipelineExecutor):
    """
    Executor to run pipelines on Slurm.
    Creates and calls a sbatch launch script.
    """

    def __init__(
        self,
        tasks: int,
        time: str,
        logging_dir: str,
        partition: str,
        cpus_per_task: int = 1,
        mem_per_cpu_gb: int = 2,
        workers: int = -1,
        job_name: str = "data_processing",
        condaenv: str = None,
        venv_path: str = None,
        sbatch_args: dict | None = None,
        max_array_size: int = 1001,
        depends: SlurmPipelineExecutor | None = None,
        **kwargs,
    ):
        """
        :param tasks: total number of tasks to run
        :param time: time limit, passed to slurm
        :param logging_dir: directory where log files and the launch script should be saved
        :param cpus_per_task: how many cpus per task
        :param job_name: slurm job name
        :param condaenv: name of a conda environment to activate before starting the job
        :param venv_path: path to a virtual environment to activate, if not using conda
        :param sbatch_args: a dictionary of other SBATCH arguments for the launch script
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.tasks = tasks
        self.workers = workers
        self.partition = partition
        self.cpus_per_task = cpus_per_task
        self.mem_per_cpu_gb = mem_per_cpu_gb
        self.logging_dir = logging_dir
        self.time = time
        self.job_name = job_name
        self.condaenv = condaenv
        self.venv_path = venv_path
        self.depends = depends
        self._sbatch_args = sbatch_args if sbatch_args else {}
        self.max_array_size = max_array_size
        self.job_id = None

    def run(self):
        if "SLURM_JOB_ID" in os.environ:
            rank = int(os.environ["SLURM_ARRAY_TASK_ID"]) + self.max_array_size * int(os.environ.get("RUN_OFFSET", 0))
            completion_file = os.path.join(self.logging_dir, "completions", f"{rank:05d}")
            if rank >= self.world_size:
                return
            if os.path.exists(completion_file):
                logger.info(f"Skipping {rank=} as it was already completed.")
                return
            stats = self._run_for_rank(rank)
            stats.save_to_disk(os.path.join(self.logging_dir, "stats", f"{rank:05d}.json"))
            open(completion_file, "w").close()
        else:
            self.launch_job()

    def launch_merge_stats(self):
        with tempfile.NamedTemporaryFile("w") as f:
            f.write(
                self.get_launch_file(
                    {
                        **self.sbatch_args,
                        "cpus-per-task": 1,
                        "mem-per-cpu": "1G",
                        "array": "0",
                        "dependency": f"afterok:{self.job_id}",
                    },
                    f'merge_stats {os.path.join(self.logging_dir, "stats")} --output {os.path.join(self.logging_dir, "stats.json")}',
                )
            )
            f.flush()
            subprocess.check_output(["sbatch", f.name])

    def launch_job(self):
        launch_script_path = os.path.join(self.logging_dir, "launch_script.slurm")
        os.makedirs(os.path.join(self.logging_dir, "stats"), exist_ok=True)
        os.makedirs(os.path.join(self.logging_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.logging_dir, "completions"), exist_ok=True)
        assert not self.depends or (
            isinstance(self.depends, SlurmPipelineExecutor) and self.depends.job_id
        ), "depends= must be a SlurmPipelineExecutor that was already launched!"

        # pickle
        with open(os.path.join(self.logging_dir, "executor.pik"), "wb") as f:
            dill.dump(self, f)

        with open(launch_script_path, "w") as f:
            f.write(self.launch_file)

        logger.info(f'Launching Slurm job {self.job_name} with launch script "{launch_script_path}"')
        run_offset = 0
        dependency = self.depends.job_id if self.depends else None
        while run_offset * self.max_array < self.tasks:
            args = ["sbatch", f"--export=ALL,RUN_OFFSET={run_offset}"]
            if dependency:
                args.append(f"--dependency=afterok:{dependency}")
            output = subprocess.check_output(args + [launch_script_path]).decode("utf-8")
            dependency = self.job_id = int(output.split()[-1])
            run_offset += 1
        logger.info(f"Slurm job launched successfully with id={self.job_id}.")
        self.launch_merge_stats()

    @property
    def max_array(self) -> int:
        return min(self.tasks, self.max_array_size) if self.max_array_size != -1 else self.tasks

    @property
    def sbatch_args(self) -> dict:
        slurm_logfile = os.path.join(self.logging_dir, "logs", "%j.out")
        return {
            "cpus-per-task": self.cpus_per_task,
            "mem-per-cpu": f"{self.mem_per_cpu_gb}G",
            "partition": self.partition,
            "job-name": self.job_name,
            "time": self.time,
            "output": slurm_logfile,
            "error": slurm_logfile,
            "array": f"0-{self.max_array - 1}{f'%{self.workers}' if self.workers != -1 else ''}",
            **self._sbatch_args,
        }

    @property
    def launch_file(self):
        return self.get_launch_file(
            self.sbatch_args,
            f'srun -l python -u -c "import dill;dill.load(open(\'{os.path.join(self.logging_dir, "executor.pik")}\', \'rb\')).run()"',
        )

    def get_launch_file(self, sbatch_args: dict, run_script: str):
        args = "\n".join([f"#SBATCH --{k}={v}" for k, v in sbatch_args.items()])

        env_command = (
            f"""conda init bash
        conda activate {self.condaenv}"""
            if self.condaenv
            else (f"source {self.venv_path}" if self.venv_path else "")
        )

        return (
            "#!/bin/bash\n"
            + args
            + textwrap.dedent(
                f"""
        echo "Starting data processing job {self.job_name}"
        {env_command}
        source ~/.bashrc
        set -xe
        {run_script}
        """
            )
        )

    @property
    def world_size(self):
        return self.tasks
