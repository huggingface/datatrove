# DataTrove

DataTrove is a library to process, filter and deduplicate text data at a very large scale. It provides a set of prebuilt commonly used processing blocks with a framework to easily add custom functionality.

DataTrove processing pipelines are platform-agnostic, running out of the box locally or on a slurm cluster. Its low memory usage and multiple step design makes it ideal for large workloads, such as to process an LLM's training data.

## Pipeline
### DataTrove Document
Each pipeline block processes data in the datatrove [`Document`](src/datatrove/data.py) format:
- `text` the actual text content for each sample
- `id` a unique id (string) for this sample
- `metadata` a dictionary where any additional info may be stored

### Types of pipeline blocks
Each pipeline block takes a generator of `Document` as input and returns another generator of `Document`.
- **readers** read data from different formats and yield `Document`
- **writers** save `Document` to disk/cloud in different formats
- **extractors** extract text content from raw formats (such as webpage html)
- **filters** filter out (remove) some `Document`s based on specific rules/criteria
- **stats** blocks to collect statistics on the dataset
- **tokens** blocks to tokenize data or count tokens
- **dedup** blocks for deduplication

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
- `tasks` total number of tasks to run
- `time` slurm time limit string. mandatory
- `workers` how many tasks to run simultaneously. If `-1`, no limit. Slurm will run `workers` tasks at a time.
- `cpus_per_task` how many cpus to give each task. `1` by default


[WIP]

## Installation

```bash
pip install -e ".[dev]"
```

Install pre-commit code style hooks:
```bash
pre-commit install
```

Run the tests:
```bash
pytest -n 4  --max-worker-restart=0 --dist=loadfile tests
```
