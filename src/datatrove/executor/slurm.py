from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from copy import deepcopy
from typing import Callable

import dill
from dill import CONTENTS_FMODE

from datatrove.executor.base import PipelineExecutor
from datatrove.io import DataFolderLike
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import get_random_str, get_timestamp, logger


def requeue_handler(signum, _frame):
    signame = signal.Signals(signum).name
    logger.warning(f"Received signal {signum} ({signame}). Requeueing and exiting...")
    subprocess.run(["scontrol", "requeue", os.environ.get("SLURM_JOB_ID")])
    sys.exit(15)


class SlurmPipelineExecutor(PipelineExecutor):
    """Execute a pipeline on a slurm cluster
    Creates and calls a sbatch launch script.

    [!] do not launch tasks from within a compute node/from another slurm task!

    Args:
        pipeline: a list of PipelineStep and/or custom functions
            with arguments (data: DocumentsPipeline, rank: int,
            world_size: int)
        tasks: total number of tasks to run the pipeline on
        time: slurm time limit
        partition: slurm partition
        cpus_per_task: how many cpus to give each task. should be 1
            except when you need to give each task more memory
        mem_per_cpu_gb: slurm option. use in conjunction with the
            above option to increase max memory
        workers: how many tasks to run simultaneously. -1 for no
            limit
        job_name: slurm job name
        env_command: command to activate a python environment, if
            needed
        condaenv: name of a conda environment to activate
        venv_path: path to a python venv to activate
        sbatch_args: dictionary with additional arguments to pass to
            sbatch
        max_array_size: the limit of tasks in a task array job on
            your slurm cluster or -1 if none. if
            tasks>max_array_size, multiple task array jobs will be
            launched
        depends: another SlurmPipelineExecutor that should run
            before this one
        depends_job_id: alternatively to the above, you can pass the job id of a dependency
        logging_dir: where to save logs, stats, etc. Should be parsable into a datatrove.io.DataFolder
        skip_completed: whether to skip tasks that were completed in
            previous runs. default: True
        slurm_logs_folder: where to store the raw slurm log files.
            must be a local path default:
            slurm_logs/$job_name/$timestamp_$randomstring
        max_array_launch_parallel: if we need multiple jobs due to max_array_size, whether to launch them all in
            one go (parallel) or sequentially
        stagger_max_array_jobs: when max_array_launch_parallel is True, this determines how many seconds to wait
            between launching each of the parallel jobs
        run_on_dependency_fail: start executing when a job we depend on finishes even if it has failed
        randomize_start_duration: the maximum number of seconds to delay the start of each task.
        requeue_signals: requeue the job and exit when one of these signals is received. Useful for when an instance
        is being reclaimed and jobs must be stopped for example. Set to None to disable
        mail_type: see https://slurm.schedmd.com/sbatch.html. Common values are (NONE, BEGIN, END, FAIL, REQUEUE, ALL)
        mail_user: email address to send notifications to
        requeue: requeue the job if it fails
        tasks_per_job: each slurm job in the job array will run these many datatrove tasks. This reduces the total nb of slurm jobs launched.
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
        qos: str = "normal",
        env_command: str = None,
        condaenv: str = None,
        venv_path: str = None,
        sbatch_args: dict | None = None,
        max_array_size: int = 1001,
        depends: SlurmPipelineExecutor | None = None,
        depends_job_id: str | None = None,
        logging_dir: DataFolderLike = None,
        skip_completed: bool = True,
        slurm_logs_folder: str = None,
        max_array_launch_parallel: bool = False,
        stagger_max_array_jobs: int = 0,
        run_on_dependency_fail: bool = False,
        randomize_start_duration: int = 0,
        requeue_signals: tuple[str] | None = ("SIGUSR1",),
        mail_type: str = "ALL",
        mail_user: str = None,
        requeue: bool = True,
        srun_args: dict = None,
        tasks_per_job: int = 1,
    ):
        super().__init__(pipeline, logging_dir, skip_completed, randomize_start_duration)
        self.tasks = tasks
        self.workers = workers
        self.partition = partition
        self.cpus_per_task = cpus_per_task
        self.mem_per_cpu_gb = mem_per_cpu_gb
        self.tasks_per_job = tasks_per_job
        self.time = time
        self.job_name = job_name
        self.qos = qos
        self.env_command = env_command
        self.condaenv = condaenv
        self.venv_path = venv_path
        self.depends = depends
        self.depends_job_id = depends_job_id
        self._sbatch_args = sbatch_args if sbatch_args else {}
        self.max_array_size = max_array_size
        self.max_array_launch_parallel = max_array_launch_parallel
        self.stagger_max_array_jobs = stagger_max_array_jobs
        self.run_on_dependency_fail = run_on_dependency_fail
        self.randomize_start_duration = randomize_start_duration
        self.job_id = None
        self.requeue_signals = requeue_signals
        self.mail_type = mail_type
        self.mail_user = mail_user
        self.srun_args = srun_args
        self.slurm_logs_folder = (
            slurm_logs_folder
            if slurm_logs_folder
            else (
                f"slurm_logs/{self.job_name}/{get_timestamp()}_{get_random_str()}"
                if not self.logging_dir.is_local()
                else self.logging_dir.resolve_paths("slurm_logs")
            )
        )
        self.requeue = requeue

    def run(self):
        """
            This method is responsible for correctly invoking `self._run_for_rank` for each task that is to be run.

            On a SlurmPipelineExecutor, we first check if we are already running on a slurm task and, if not, we launch it.
        Returns:

        """
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            # we are already "inside" the slurm task, get our rank from env vars and run pipeline
            slurm_rank = int(os.environ["SLURM_ARRAY_TASK_ID"]) + self.max_array_size * int(
                os.environ.get("RUN_OFFSET", 0)
            )
            ranks_to_run_range = (slurm_rank * self.tasks_per_job, (slurm_rank + 1) * self.tasks_per_job)
            with self.logging_dir.open("ranks_to_run.json", "r") as ranks_to_run_file:
                all_ranks = json.load(ranks_to_run_file)
            if ranks_to_run_range[0] >= len(all_ranks):
                return

            for ss in self.requeue_signals or []:
                signal.signal(signal.Signals[ss], requeue_handler)

            for rank_to_run in range(*ranks_to_run_range):
                if rank_to_run >= len(all_ranks):
                    break
                rank = all_ranks[rank_to_run]

                self._run_for_rank(rank)
        else:
            # we still have to launch the job
            self.launch_job()

    def launch_merge_stats(self):
        """
            Launch a slurm task to merge the stats of each individual task into one big stats summary file.
        Returns:

        """
        launch_slurm_job(
            self.get_launch_file_contents(
                {
                    **self.get_sbatch_args(),
                    "cpus-per-task": 1,
                    "mem-per-cpu": "1G",
                    "dependency": f"afterok:{self.job_id}",
                },
                f'merge_stats {self.logging_dir.resolve_paths("stats")} '
                f'-o {self.logging_dir.resolve_paths("stats.json")}',
            )
        )

    @property
    def dependency(self) -> str:
        """
            Resolve list of jobs we depend on and return it as a slurm string.
        Returns:

        """
        dependency = []
        if self.depends_job_id:
            dependency.append(f"{'afterany' if self.run_on_dependency_fail else 'afterok'}:" f"{self.depends_job_id}")
        if self.job_id and not self.max_array_launch_parallel:
            dependency.append(f"afterany:{self.job_id}")
        return ",".join(dependency)

    def launch_job(self):
        """
            Takes care of creating a sbatch script for this pipeline and launching it.
        Returns:

        """
        assert not self.depends or (
            isinstance(self.depends, SlurmPipelineExecutor)
        ), "depends= must be a SlurmPipelineExecutor"
        if self.depends:
            # take care of launching any unlaunched dependencies and getting their slurm job ids
            if not self.depends.job_id:
                logger.info(f'Launching dependency job "{self.depends.job_name}"')
                self.depends.launch_job()
            if self.depends.job_id != -1:
                self.depends_job_id = self.depends.job_id
            self.depends = None  # avoid pickling the entire dependency and possibly its dependencies

        ranks_to_run = self.get_incomplete_ranks()
        if len(ranks_to_run) == 0:
            logger.info(f"Skipping launch of {self.job_name} as all {self.tasks} tasks have already been completed.")
            self.job_id = -1
            return

        executor = deepcopy(self)

        # pickle. The slurm job will load the executor from this pik file
        with self.logging_dir.open("executor.pik", "wb") as executor_f:
            dill.dump(executor, executor_f, fmode=CONTENTS_FMODE)
        self.save_executor_as_json()

        with self.logging_dir.open("ranks_to_run.json", "w") as ranks_to_run_file:
            # we actually save this (only once) to avoid race conditions
            json.dump(ranks_to_run, ranks_to_run_file)

        nb_jobs_to_launch = math.ceil(len(ranks_to_run) / self.tasks_per_job)
        max_array = min(nb_jobs_to_launch, self.max_array_size) if self.max_array_size != -1 else nb_jobs_to_launch

        # create the actual sbatch script
        srun_args_str = " ".join([f"--{k}={v}" for k, v in self.srun_args.items()]) if self.srun_args else ""
        launch_file_contents = self.get_launch_file_contents(
            self.get_sbatch_args(max_array),
            f"srun {srun_args_str} -l launch_pickled_pipeline {self.logging_dir.resolve_paths('executor.pik')}",
        )
        # save it
        with self.logging_dir.open("launch_script.slurm", "w") as launchscript_f:
            launchscript_f.write(launch_file_contents)
        logger.info(
            f"Launching Slurm job {self.job_name} ({len(ranks_to_run)} tasks) with launch script "
            f'"{self.logging_dir.resolve_paths("launch_script.slurm")}"'
        )

        # launch (possibly multiple) jobs
        launched_jobs = 0
        while launched_jobs * max_array < nb_jobs_to_launch:
            if launched_jobs and self.max_array_launch_parallel and self.stagger_max_array_jobs > 0:
                time.sleep(self.stagger_max_array_jobs)
            args = [f"--export=ALL,RUN_OFFSET={launched_jobs}"]
            if self.dependency:
                args.append(f"--dependency={self.dependency}")
            self.job_id = launch_slurm_job(launch_file_contents, *args)
            launched_jobs += 1
        logger.info(f"Slurm job launched successfully with (last) id={self.job_id}.")
        self.launch_merge_stats()

    def get_sbatch_args(self, max_array: int = 1) -> dict:
        """
            Get a dictionary with all the sbatch directives we want to include
        Args:
            max_array: max array size

        Returns: a dictionary with all the sbatch directives

        """
        # this one we actually have to create as slurm will be writing here
        os.makedirs(self.slurm_logs_folder, exist_ok=True)
        slurm_logfile = os.path.join(self.slurm_logs_folder, "%A_%a.out")
        sbatch_args = {
            "cpus-per-task": self.cpus_per_task,
            "mem-per-cpu": f"{self.mem_per_cpu_gb}G",
            "partition": self.partition,
            "job-name": self.job_name,
            "time": self.time,
            "output": slurm_logfile,
            "error": slurm_logfile,
            "array": f"0-{max_array - 1}{f'%{self.workers}' if self.workers != -1 else ''}",
            **({"mail-type": self.mail_type, "mail-user": self.mail_user} if self.mail_user else {}),
            **self._sbatch_args,
        }
        if self.requeue:
            sbatch_args["requeue"] = ""
        if self.qos:
            sbatch_args["qos"] = self.qos
        return sbatch_args

    def get_launch_file_contents(self, sbatch_args: dict, run_script: str) -> str:
        """
            Actually generate the sbatch script
        Args:
            sbatch_args: dictionary with all the sbatch directives to include
            run_script: command to be invoked by this slurm job

        Returns:

        """
        args = "\n".join([f"#SBATCH --{k}={v}" if v else f"#SBATCH --{k}" for k, v in sbatch_args.items()])

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
        export PYTHONUNBUFFERED=TRUE
        {run_script}
        """
            )
        )

    @property
    def world_size(self) -> int:
        return self.tasks


def launch_slurm_job(launch_file_contents, *args):
    """
        Small helper function to save a sbatch script and call it.
    Args:
        launch_file_contents: Contents of the sbatch script
        *args: any other arguments to pass to the sbatch command

    Returns: the id of the launched slurm job

    """
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(launch_file_contents)
        f.flush()
        return subprocess.check_output(["sbatch", *args, f.name]).decode("utf-8").split()[-1]
