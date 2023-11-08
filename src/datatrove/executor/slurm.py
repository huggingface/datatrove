from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
from typing import Callable

import dill
from loguru import logger

from datatrove.executor.base import PipelineExecutor
from datatrove.io import BaseOutputDataFolder, S3OutputDataFolder
from datatrove.pipeline.base import PipelineStep


class SlurmPipelineExecutor(PipelineExecutor):
    """
    Executor to run pipelines on Slurm.
    Creates and calls a sbatch launch script.
    """

    def __init__(
        self,
        pipeline: list[PipelineStep | Callable],
        tasks: int,
        time: str,
        partition: str,
        cpus_per_task: int = 1,
        mem_per_cpu_gb: int = 2,
        workers: int = -1,
        job_name: str = "data_processing",
        env_command: str = None,
        condaenv: str = None,
        venv_path: str = None,
        sbatch_args: dict | None = None,
        max_array_size: int = 1001,
        depends: SlurmPipelineExecutor | None = None,
        logging_dir: BaseOutputDataFolder = None,
    ):
        """
        :param tasks: total number of tasks to run
        :param time: time limit, passed to slurm
        :param cpus_per_task: how many cpus per task
        :param job_name: slurm job name
        :param condaenv: name of a conda environment to activate before starting the job
        :param venv_path: path to a virtual environment to activate, if not using conda
        :param sbatch_args: a dictionary of other SBATCH arguments for the launch script
        :param kwargs:
        """
        if isinstance(logging_dir, S3OutputDataFolder):
            logging_dir.cleanup = False  # if the files are removed from disk job launch will fail
        super().__init__(pipeline, logging_dir)
        self.tasks = tasks
        self.workers = workers
        self.partition = partition
        self.cpus_per_task = cpus_per_task
        self.mem_per_cpu_gb = mem_per_cpu_gb
        self.time = time
        self.job_name = job_name
        self.env_command = env_command
        self.condaenv = condaenv
        self.venv_path = venv_path
        self.depends = depends
        self._sbatch_args = sbatch_args if sbatch_args else {}
        self.max_array_size = max_array_size
        self.job_ids = []
        self.depends_job_ids = []
        self.launched = False

    def run(self):
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            rank = int(os.environ["SLURM_ARRAY_TASK_ID"]) + self.max_array_size * int(os.environ.get("RUN_OFFSET", 0))
            if rank >= self.world_size:
                return
            self._run_for_rank(rank)
            self.logging_dir.close()  # make sure everything is properly saved (logs etc)
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
                        "dependency": f"afterok:{','.join(self.job_ids)}",
                    },
                    f'merge_stats {os.path.join(self.logging_dir.path, "stats")} '
                    f'--output {os.path.join(self.logging_dir.path, "stats.json")}',
                )
            )
            f.flush()
            subprocess.check_output(["sbatch", f.name])

    @property
    def dependency(self):
        dependency = []
        if self.depends_job_ids:
            dependency.append(f"afterok:{','.join(self.depends_job_ids)}")
        if self.job_ids:
            dependency.append(f"afterany:{self.job_ids[-1]}")
        return ",".join(dependency)

    def launch_job(self):
        assert not self.depends or (
            isinstance(self.depends, SlurmPipelineExecutor)
        ), "depends= must be a SlurmPipelineExecutor"
        if self.depends:
            if not self.depends.launched:
                logger.info(f'Launching dependency job "{self.depends.job_name}"')
                self.depends.launch_job()
            self.depends_job_ids = self.depends.job_ids
            self.depends = None  # avoid pickling the entire dependency and possibly its dependencies

        if all(map(self.is_rank_completed, range(self.tasks))):
            logger.info(f"Skipping launch of {self.job_name} as all {self.tasks} tasks have already been completed.")
            self.launched = True
            return

        # pickle
        with self.logging_dir.open("executor.pik", "wb") as executor_f:
            dill.dump(self, executor_f)

        with self.logging_dir.open("launch_script.slurm") as launchscript_f:
            launchscript_f.write(
                self.get_launch_file(
                    self.sbatch_args,
                    f"srun -l python -u -c \"import dill;dill.load(open('{executor_f.persistent_local_path}', 'rb')).run()\"",
                )
            )
        logger.info(f'Launching Slurm job {self.job_name} with launch script "{launchscript_f.path}"')

        self.job_ids = []
        while len(self.job_ids) * self.max_array < self.tasks:
            args = ["sbatch", f"--export=ALL,RUN_OFFSET={len(self.job_ids)}"]
            if self.dependency:
                args.append(f"--dependency={self.dependency}")
            output = subprocess.check_output(args + [launchscript_f.persistent_local_path]).decode("utf-8")
            self.job_ids.append(output.split()[-1])
        self.launched = True
        logger.info(f"Slurm job launched successfully with id(s)={','.join(self.job_ids)}.")
        self.launch_merge_stats()
        self.logging_dir.close()

    @property
    def max_array(self) -> int:
        return min(self.tasks, self.max_array_size) if self.max_array_size != -1 else self.tasks

    @property
    def sbatch_args(self) -> dict:
        slurm_logfile = self.logging_dir.open("slurm_logs/%j.out")
        return {
            "cpus-per-task": self.cpus_per_task,
            "mem-per-cpu": f"{self.mem_per_cpu_gb}G",
            "partition": self.partition,
            "job-name": self.job_name,
            "time": self.time,
            "output": slurm_logfile.persistent_local_path,
            "error": slurm_logfile.persistent_local_path,
            "array": f"0-{self.max_array - 1}{f'%{self.workers}' if self.workers != -1 else ''}",
            **self._sbatch_args,
        }

    def get_launch_file(self, sbatch_args: dict, run_script: str):
        args = "\n".join([f"#SBATCH --{k}={v}" for k, v in sbatch_args.items()])

        env_command = (
            self.env_command
            if self.env_command
            else (
                f"""conda init bash
        conda activate {self.condaenv}
        source ~/.bashrc"""
                if self.condaenv
                else (f"source {self.venv_path}" if self.venv_path else "")
            )
        )

        return (
            "#!/bin/bash\n"
            + args
            + textwrap.dedent(
                f"""
        echo "Starting data processing job {self.job_name}"
        {env_command}
        set -xe
        {run_script}
        """
            )
        )

    @property
    def world_size(self):
        return self.tasks
