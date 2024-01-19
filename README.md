# DataTrove

DataTrove is a library to process, filter and deduplicate text data at a very large scale. It provides a set of prebuilt commonly used processing blocks with a framework to easily add custom functionality.

DataTrove processing pipelines are platform-agnostic, running out of the box locally or on a slurm cluster. Its (relatively) low memory usage and multiple step design makes it ideal for large workloads, such as to process an LLM's training data.

Local, remote and other file systems are supported through [fsspec](https://filesystem-spec.readthedocs.io/en/latest/).

## Installation

```bash
git clone git@github.com:huggingface/datatrove.git && cd datatrove
pip install -e ".[FLAVOUR]"
```
Available flavours (combine them with `,` i.e. `[processing,s3]`:
- `all` installs everything
- `io` dependencies to read `warc/arc/wet` files and arrow/parquet formats
- `processing` dependencies for text extraction, filtering and tokenization
- `s3` s3 support
- `cli` for command line tools

## Pipeline
### DataTrove Document
Each pipeline block processes data in the datatrove [`Document`](src/datatrove/data.py) format:
- `text` the actual text content for each sample
- `id` a unique id (string) for this sample
- `metadata` a dictionary where any additional info may be stored

### Types of pipeline blocks
Each pipeline block takes a generator of `Document` as input and returns another generator of `Document`.
- **[readers](src/datatrove/pipeline/readers)** read data from different formats and yield `Document`
- **[writers](src/datatrove/pipeline/writers)** save `Document` to disk/cloud in different formats
- **[extractors](src/datatrove/pipeline/extractors)** extract text content from raw formats (such as webpage html)
- **[filters](src/datatrove/pipeline/filters)** filter out (remove) some `Document`s based on specific rules/criteria
- **[stats](src/datatrove/pipeline/stats)** blocks to collect statistics on the dataset
- **[tokens](src/datatrove/pipeline/tokens)** blocks to tokenize data or count tokens
- **[dedup](src/datatrove/pipeline/dedup)** blocks for deduplication
### Full pipeline
A pipeline is defined as a list of pipeline blocks. As an example, the following pipeline would read data from disk, randomly filter (remove) some documents and write them back to disk:
```python
from datatrove.pipeline.readers import CSVReader
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import JsonlWriter

pipeline = [
    CSVReader(
        data_folder="/my/input/path"
    ),
    SamplerFilter(rate=0.5),
    JsonlWriter(
        output_folder="/my/output/path"
    )
]
```

## Executors
Pipelines are platform-agnostic, which means that the same pipeline can smoothly run on different execution environments without any changes to its steps. Each environment has its own PipelineExecutor.
Some options common to all executors:
- `pipeline` a list consisting of the pipeline steps that should be run
- `logging_dir` a datafolder where log files, statistics and more should be saved
- `skip_completed` (_bool_, `True` by default) datatrove keeps track of completed tasks so that when you relaunch a job they can be skipped. Set this to `False` to disable this behaviour

Call an executor's `run` method to execute its pipeline.


### LocalPipelineExecutor
This executor will launch a pipeline on a local machine.
Options:
- `tasks` total number of tasks to run
- `workers` how many tasks to run simultaneously. If `-1`, no limit. Anything `> 1` will use multiprocessing to execute the tasks.
- `start_method` method to use to spawn a multiprocessing Pool. Ignored if `workers` is 1

<details>
  <summary>Example executor</summary>

```python
from datatrove.executor import LocalPipelineExecutor
executor = LocalPipelineExecutor(
    pipeline=[
        ...
    ],
    logging_dir="logs/",
    tasks=10,
    workers=5
)
executor.run()
```
</details>

### SlurmPipelineExecutor
This executor will launch a pipeline on a slurm cluster, using slurm job arrays to group and manage tasks.
Options:
- `tasks` total number of tasks to run. **required**
- `time` slurm time limit string. **required**
- `partition` slurm partition. **required**
- `workers` how many tasks to run simultaneously. If `-1`, no limit. Slurm will run `workers` tasks at a time. (default: `-1`)
- `job_name` slurm job name (default: "data_processing)
- `depends` another SlurmPipelineExecutor instance, which will be a dependency of this pipeline (current pipeline will only start executing after the depended on pipeline successfully completes)
- `sbatch_args` dictionary with any other arguments you would like to pass to sbatch
- `slurm_logs_folder` where to save the slurm log files. If using a local path for `logging_dir`, they will be saved on `logging_dir/slurm_logs`. If not, they will be saved as a subdir of the current directory.
<details>
  <summary>Other options</summary>

- `cpus_per_task` how many cpus to give each task (default: `1`)
- `qos` slurm qos (default: "normal")
- `mem_per_cpu_gb` memory per cpu, in GB (default: 2)
- `env_command` custom command to activate a python environment, if needed
- `condaenv` conda environment to activate
- `venv_path` path to a python environment to activate
- `max_array_size` the _MaxArraySize_ value in `$ scontrol show config`. If number of tasks exceeds this number, it will split into multiple array jobs (default: 1001)
- `max_array_launch_parallel` if we need multiple jobs due to max_array_size, whether to launch them all in one go (parallel) or sequentially (default: `False`)
- `stagger_max_array_jobs` when max_array_launch_parallel is True, this determines how many seconds to wait between launching each of the parallel jobs (default: `0`)
- `run_on_dependency_fail` start executing when a job we depend on finishes even if it has failed (default: `False`)
- `randomize_start` randomize the start of each task in a job in a ~3 min window. Useful when heavily hitting an s3 bucket for example. (default: `False`)
</details>
<details>
  <summary>Example executor</summary>

```python
from datatrove.executor import SlurmPipelineExecutor
executor1 = SlurmPipelineExecutor(
    pipeline=[
        ...
    ],
    job_name="my_cool_job1",
    logging_dir="logs/job1",
    tasks=500,
    workers=100,  # omit to run all at once
    time="10:00:00",  # 10 hours
    partition="hopper-cpu"
)
executor2 = SlurmPipelineExecutor(
    pipeline=[
        ...
    ],
    job_name="my_cool_job2",
    logging_dir="logs/job2",
    tasks=1,
    time="5:00:00",  # 5 hours
    partition="hopper-cpu",
    depends=executor1  # this pipeline will only be launched after executor1 successfuly completes
)
# executor1.run()
executor2.run() # this will actually launch executor1, as it is a dependency, so no need to launch it explicitly
```
</details>

## Logging
For a pipeline with `logging_dir` **mylogspath/exp1**, the following folder structure would be created:

<details>
  <summary>See folder structure</summary>

```
└── mylogspath/exp1
    │── executor.json ⟵ json dump of the executor options and pipeline steps
    │── launch_script.slurm ⟵ the slurm config created and used to launch this job (if running on slurm)
    │── executor.pik ⟵ the slurm config created and used to launch this job (if running on slurm)
    │── ranks_to_run.json ⟵ list of tasks that are being run
    │── logs/
    │   └──[task_00000.log, task_00001.log, task_00002.log, ...] ⟵ individual logging files for each task
    │── completions/
    │   └──[00004, 00007, 00204, ...] ⟵ empty files marking a task as completed. Using when relaunching/resuming a job (only unfinished tasks will be run)
    │── stats/
    │   └──[00000.json, 00001.json, 00002.json, ...] ⟵ individual stats for each task (number of samples processed, filtered, removed, etc)
    └── stats.json ⟵ global stats from all tasks
```
</details>

## Practical guides


## Contributing

```bash
pip install -e ".[dev]"
```

Install pre-commit code style hooks:
```bash
pre-commit install
```

Run the tests:
```bash
pytest -sv ./tests/ 
```
