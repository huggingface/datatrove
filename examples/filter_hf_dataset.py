"""
This file contains code to:
1 - Load a parquet-format Hugging Face dataset from the hub.
2 - Filter the dataset (to include only entries that contain the word 'hugging' in the text column).
3 - Push the filtered dataset back to the hub.
"""

import argparse


parser = argparse.ArgumentParser("Filter an HF dataset and push the result to the hub")

parser.add_argument("input_dataset", type=str, help="HF dataset to filter")
parser.add_argument("output_name", type=str, help="Name of the output dataset")
parser.add_argument("--n_tasks", type=int, help="number of tasks", default=100)
parser.add_argument("--text_key", type=int, help="text column", default="text")

ORG_NAME = "my_org"
LOCAL_PATH = "my_local_path"
LOCAL_LOGS_PATH = "my_local_logs_path"

if __name__ == "__main__":
    args = parser.parse_args()
    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers.huggingface import HuggingFaceDatasetWriter

    dist_executor = SlurmPipelineExecutor(
        job_name=f"filter-{args.output_name}",
        pipeline=[
            ParquetReader(args.input_dataset, glob_pattern="**/*.parquet", text_key=args.text_key),
            LambdaFilter(lambda doc: "hugging" in doc.text),  # add your custom filter here
            HuggingFaceDatasetWriter(
                dataset=f"{ORG_NAME}/{args.output_name}",
                private=True,
                local_working_dir=f"{LOCAL_PATH}/{args.output_name}",
                output_filename="data/${rank}.parquet",
                cleanup=True,
            ),
        ],
        tasks=args.n_tasks,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir=f"{LOCAL_LOGS_PATH}/{args.output_name}",
        cpus_per_task=12,
        qos="high",
        mem_per_cpu_gb=3,
    )
    dist_executor.run()
