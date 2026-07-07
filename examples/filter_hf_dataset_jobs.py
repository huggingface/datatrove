"""
This file contains code to:
1 - Load a parquet-format Hugging Face dataset from the hub.
2 - Filter the dataset (to include only entries that contain the word 'hugging' in the text column).
3 - Push the filtered dataset back to the hub.

...running on Hugging Face Jobs. It is the `JobsPipelineExecutor` twin of `filter_hf_dataset.py`
(which uses Slurm) — the pipeline is identical; only the executor changes. Your machine just
orchestrates: the filtering runs across a pool of Jobs that coordinate through a shared, *remote*
`logging_dir`.

Differences from the Slurm version (see comments inline):
- `logging_dir` MUST be remote (hf:// or s3://) — the launcher and the Jobs are different machines.
- `dependencies` are declared explicitly (each Job builds a fresh `uv` env); there is no
  env_command / conda / venv to pre-activate.
- Slurm's partition / qos / cpus_per_task / mem_per_cpu_gb / time collapse into one `flavor` (+ `timeout`).
- `workers` caps concurrent Jobs (Slurm's `%workers`).
"""

import argparse


parser = argparse.ArgumentParser("Filter an HF dataset on HF Jobs and push the result to the hub")

parser.add_argument("input_dataset", type=str, help="HF dataset to filter")
parser.add_argument("output_name", type=str, help="Name of the output dataset")
parser.add_argument("--n_tasks", type=int, help="number of tasks", default=100)
parser.add_argument("--workers", type=int, help="max concurrent Jobs (-1 = all at once)", default=20)
parser.add_argument("--flavor", type=str, help="HF Jobs hardware flavor", default="cpu-basic")
parser.add_argument("--text_key", type=str, help="text column", default="text")

ORG_NAME = "my_org"
# Jobs coordinate ONLY through a shared REMOTE logging_dir (hf:// or s3://), never a local path.
# Keep this OUT of the output dataset repo — otherwise the Hub dataset viewer tries to cast
# executor.json / stats.json into the same table as your parquet and errors.
REMOTE_LOGS_PATH = "hf://datasets/my_org/my-filter-logs"
# datatrove + every extra/dep your pipeline steps need, installed into each Job's uv env.
# NB: even LambdaFilter needs the `processing` extra — importing datatrove.pipeline.filters
# eagerly pulls in modules that require `regex`. Match the extras to the steps you use.
# Until JobsPipelineExecutor is released, install datatrove from your branch, e.g.:
#   "datatrove[io,processing] @ git+https://github.com/<user>/datatrove@<branch>"
DEPENDENCIES = ["datatrove[io,processing]"]

if __name__ == "__main__":
    args = parser.parse_args()
    from datatrove.executor import JobsPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers.huggingface import HuggingFaceDatasetWriter

    dist_executor = JobsPipelineExecutor(
        job_name=f"filter-{args.output_name}",
        pipeline=[
            ParquetReader(args.input_dataset, glob_pattern="**/*.parquet", text_key=args.text_key),
            LambdaFilter(lambda doc: "hugging" in doc.text),  # add your custom filter here
            HuggingFaceDatasetWriter(
                dataset=f"{ORG_NAME}/{args.output_name}",
                private=True,
                local_working_dir=f"/tmp/{args.output_name}",  # Job-local scratch (ephemeral); cleaned up
                output_filename="data/${rank}.parquet",
                cleanup=True,
            ),
        ],
        tasks=args.n_tasks,
        workers=args.workers,  # concurrency cap — the analog of Slurm's `--array 0-M%workers`
        logging_dir=f"{REMOTE_LOGS_PATH}/{args.output_name}",  # remote (hf://) — shared by launcher + Jobs
        flavor=args.flavor,  # one hardware knob instead of partition/qos/cpus_per_task/mem_per_cpu_gb
        dependencies=DEPENDENCIES,  # each Job builds a fresh uv env — declare deps explicitly
        timeout="2h",
    )
    dist_executor.run()
