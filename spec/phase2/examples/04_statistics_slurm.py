from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.stats import DocStats, LangStats

pipeline = [
    # Use multiple C4 files so both nodes get work
    JsonlReader("hf://datasets/allenai/c4/en/",
                glob_pattern="c4-train.0000[0-3]-of-01024.json.gz",  # 4 files
                limit=200),  # 200 docs per file = 800 total
    DocStats(
        output_folder="/tmp/stats_truly_distributed/",
        histogram_round_digits=1,
    ),
    LangStats(
        output_folder="/tmp/stats_truly_distributed/",
        language="en",
    ),
]

executor = SlurmPipelineExecutor(
    job_name="true_distributed",
    pipeline=pipeline,
    tasks=2,  # 2 tasks, should get 2 files each
    time="00:10:00",
    partition="gpu",
    logging_dir="/tmp/logs_truly_distributed/",
    cpus_per_task=8,
    mem_per_cpu_gb=8,
)

executor.run()