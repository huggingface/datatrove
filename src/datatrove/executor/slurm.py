from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import textwrap
import time
from copy import deepcopy
from typing import Callable

import dill
from dill import CONTENTS_FMODE
from fsspec.implementations.local import LocalFileSystem
from loguru import logger

from datatrove.executor.base import PipelineExecutor
from datatrove.io import DataFolderLike
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import get_random_str, get_timestamp


class SlurmPipelineExecutor(PipelineExecutor):
    """Executor to run pipelines on Slurm.
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
        qos: str = "normal",
        env_command: str = None,
        condaenv: str = None,
        venv_path: str = None,
        sbatch_args: dict | None = None,
        max_array_size: int = 1001,
        depends: SlurmPipelineExecutor | None = None,
        logging_dir: DataFolderLike = None,
        skip_completed: bool = True,
        slurm_logs_folder: str = None,
        max_array_launch_parallel: bool = False,
        stagger_max_array_jobs: int = 0,
        run_on_dependency_fail: bool = False,
        randomize_start: bool = False,
        requeue_signals: tuple[str] | None = ("SIGUSR1",),
    ):
        """Execute a pipeline on a slurm cluster

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
            randomize_start: randomize the start of each task in a job in a ~3 min window
            requeue_signals: requeue the job and exit when this signal is received. Useful for when an instance is
            being reclaimed and jobs must be stopped for example
        """
        super().__init__(pipeline, logging_dir, skip_completed)
        self.tasks = tasks
        self.workers = workers
        self.partition = partition
        self.cpus_per_task = cpus_per_task
        self.mem_per_cpu_gb = mem_per_cpu_gb
        self.time = time
        self.job_name = job_name
        self.qos = qos
        self.env_command = env_command
        self.condaenv = condaenv
        self.venv_path = venv_path
        self.depends = depends
        self._sbatch_args = sbatch_args if sbatch_args else {}
        self.max_array_size = max_array_size
        self.max_array_launch_parallel = max_array_launch_parallel
        self.stagger_max_array_jobs = stagger_max_array_jobs
        self.run_on_dependency_fail = run_on_dependency_fail
        self.randomize_start = randomize_start
        self.job_id = None
        self.depends_job_id = None
        self.requeue_signals = requeue_signals
        self.slurm_logs_folder = (
            slurm_logs_folder
            if slurm_logs_folder
            else (
                f"slurm_logs/{self.job_name}/{get_timestamp()}_{get_random_str()}"
                if not isinstance(self.logging_dir.fs, LocalFileSystem)
                else self.logging_dir.resolve_paths("slurm_logs")
            )
        )

    def run(self):
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            slurm_rank = int(os.environ["SLURM_ARRAY_TASK_ID"]) + self.max_array_size * int(
                os.environ.get("RUN_OFFSET", 0)
            )
            with self.logging_dir.open("ranks_to_run.json", "r") as ranks_to_run_file:
                all_ranks = json.load(ranks_to_run_file)
            if slurm_rank >= len(all_ranks):
                return
            rank = all_ranks[slurm_rank]
            if self.randomize_start:
                time.sleep(random.randint(0, 60 * 3))
            self._run_for_rank(rank)
        else:
            self.launch_job()

    def launch_merge_stats(self):
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
    def dependency(self):
        dependency = []
        if self.depends_job_id:
            dependency.append(f"{'afterany' if self.run_on_dependency_fail else 'afterok'}:" f"{self.depends_job_id}")
        if self.job_id and not self.max_array_launch_parallel:
            dependency.append(f"afterany:{self.job_id}")
        return ",".join(dependency)

    def launch_job(self):
        assert not self.depends or (
            isinstance(self.depends, SlurmPipelineExecutor)
        ), "depends= must be a SlurmPipelineExecutor"
        if self.depends:
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

        # pickle
        with self.logging_dir.open("executor.pik", "wb") as executor_f:
            dill.dump(executor, executor_f, fmode=CONTENTS_FMODE)
        self.save_executor_as_json()

        with self.logging_dir.open("ranks_to_run.json", "w") as ranks_to_run_file:
            # we actually save this (only once) to avoid race conditions
            json.dump(ranks_to_run, ranks_to_run_file)

        max_array = min(len(ranks_to_run), self.max_array_size) if self.max_array_size != -1 else len(ranks_to_run)

        launch_file_contents = self.get_launch_file_contents(
            self.get_sbatch_args(max_array),
            f"srun -l launch_pickled_pipeline {self.logging_dir.resolve_paths('executor.pik')}",
        )
        with self.logging_dir.open("launch_script.slurm", "w") as launchscript_f:
            launchscript_f.write(launch_file_contents)
        logger.info(
            f"Launching Slurm job {self.job_name} ({len(ranks_to_run)} tasks) with launch script "
            f'"{self.logging_dir.resolve_paths("launch_script.slurm")}"'
        )

        launched_jobs = 0
        while launched_jobs * max_array < len(ranks_to_run):
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
        # this one we actually have to create as slurm will be writing here
        os.makedirs(self.slurm_logs_folder, exist_ok=True)
        slurm_logfile = os.path.join(self.slurm_logs_folder, "%A_%a.out")
        return {
            "cpus-per-task": self.cpus_per_task,
            "mem-per-cpu": f"{self.mem_per_cpu_gb}G",
            "partition": self.partition,
            "job-name": self.job_name,
            "time": self.time,
            "output": slurm_logfile,
            "error": slurm_logfile,
            "array": f"0-{max_array - 1}{f'%{self.workers}' if self.workers != -1 else ''}",
            "requeue": "",
            "qos": self.qos,
            **self._sbatch_args,
        }

    def get_launch_file_contents(self, sbatch_args: dict, run_script: str):
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

        trap_signals = (
            "trap 'scontrol requeue ${SLURM_JOB_ID}; exit 15' " + " ".join([self.requeue_signals])
            if self.requeue_signals
            else ""
        )

        return (
            "#!/bin/bash\n"
            + args
            + textwrap.dedent(
                f"""
        {trap_signals}
        echo "Starting data processing job {self.job_name}"
        {env_command}
        set -xe
        export PYTHONUNBUFFERED=TRUE
        {run_script}
        """
            )
        )

    @property
    def world_size(self):
        return self.tasks


def launch_slurm_job(launch_file_contents, *args):
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(launch_file_contents)
        f.flush()
        return subprocess.check_output(["sbatch", *args, f.name]).decode("utf-8").split()[-1]
