import argparse
import dataclasses

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters.sampler_filter import SamplerFilter
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.stats import DocStats, LineStats, StatsMerger, TopKConfig, WordStats


TOTAL_TASKS = 500

parser = argparse.ArgumentParser(description="Summary Stats")
parser.add_argument("dump_path", help="Dump name sampler")
parser.add_argument("sample_rate", type=float, help="Sample rate")
parser.add_argument("--prefix", default="", help="Prefix")
parser.add_argument("--glob", help="Glob pattern")
parser.add_argument("--text_key", default="text", help="Text key")
parser.add_argument("--reader", default="jsonl", help="Reader type")

if __name__ == "__main__":
    args = parser.parse_args()
    experiment_name = args.dump_path.replace("/", "_")
    LOCAL_LOGS_FOLDER = f"/logs/{experiment_name}"
    DATA_FOLDER = f"s3://data/{experiment_name}"
    SOURCE = f"{args.prefix}/{args.dump_path}"
    print(SOURCE)

    top_k_config = TopKConfig(top_k_groups=["fqdn", "suffix"], top_k=10_000)

    compute = SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(SOURCE, doc_progress=True, limit=-1, glob_pattern=args.glob, text_key=args.text_key),
            # Sampling is fine for summary stats
            SamplerFilter(
                rate=args.sample_rate,
            ),
            WordStats(
                output_folder=DATA_FOLDER,
                top_k_config=top_k_config,
            ),
            LineStats(
                output_folder=DATA_FOLDER,
                top_k_config=top_k_config,
            ),
            DocStats(
                output_folder=DATA_FOLDER,
                top_k_config=top_k_config,
            ),
        ],
        tasks=TOTAL_TASKS,
        job_name=f"summary-stats-{experiment_name}",
        time="24:00:00",
        partition="hopper-cpu",
        logging_dir=f"{LOCAL_LOGS_FOLDER}-compute",
        qos="normal",
        mem_per_cpu_gb=2,
        cpus_per_task=1,
    )

    merger = SlurmPipelineExecutor(
        pipeline=[
            StatsMerger(
                input_folder=DATA_FOLDER,
                output_folder=f"{DATA_FOLDER}",
                remove_input=False,
                top_k_config=dataclasses.replace(top_k_config, top_k=8_000),
            ),
        ],
        tasks=TOTAL_TASKS,
        job_name=f"merging-stats-{experiment_name}",
        time="24:00:00",
        partition="hopper-cpu",
        logging_dir=f"{LOCAL_LOGS_FOLDER}-merge",
        qos="normal",
        mem_per_cpu_gb=2,
        cpus_per_task=1,
        depends=compute,
    )

    merger.run()
