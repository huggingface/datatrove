import os
import subprocess
import sys
import textwrap

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
        self._sbatch_args = sbatch_args if sbatch_args else {}

    def run(self):
        if "SLURM_JOB_ID" in os.environ:
            rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
            self._run_for_rank(rank)
        else:
            self.launch_job()

    def launch_job(self):
        launch_script_path = os.path.join(self.logging_dir, "launch_script.slurm")
        os.makedirs(self.logging_dir, exist_ok=True)
        with open(launch_script_path, "w") as f:
            f.write(self.launch_file)

        logger.info(f'Launching Slurm job {self.job_name} with launch script "{launch_script_path}"')
        output = subprocess.check_output(["sbatch", launch_script_path]).decode("utf-8")
        job_id = int(output.split()[-1])
        logger.info(f"Slurm job launched successfully with id={job_id}.")

    @property
    def sbatch_args(self) -> dict:
        slurm_logfile = os.path.join(self.logging_dir, "%j.out")
        return {
            "cpus-per-task": self.cpus_per_task,
            "mem-per-cpu": f"{self.mem_per_cpu_gb}G",
            "partition": self.partition,
            "job-name": self.job_name,
            "time": self.time,
            "output": slurm_logfile,
            "error": slurm_logfile,
            "array": f"0-{self.tasks - 1}{f'%{self.workers}' if self.workers != -1 else ''}",
            **self._sbatch_args,
        }

    @property
    def launch_file(self):
        args = "\n".join([f"#SBATCH --{k}={v}" for k, v in self.sbatch_args.items()])

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
        srun -l python -u {os.path.abspath(sys.argv[0])}
        """
            )
        )

    @property
    def world_size(self):
        return self.tasks
