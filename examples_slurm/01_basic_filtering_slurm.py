from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LambdaFilter, SamplerFilter
from datatrove.pipeline.writers import JsonlWriter

pipeline = [
    JsonlReader("hf://datasets/allenai/c4/en/",
                glob_pattern="c4-train.00000-of-01024.json.gz",
                limit=100),
    LambdaFilter(lambda doc: len(doc.text) > 100),
    LambdaFilter(lambda doc: any(keyword in doc.text.lower()
                for keyword in ["data", "learning", "computer", "science"])),
    SamplerFilter(rate=0.5),
    JsonlWriter("/tmp/output/")
]

executor = SlurmPipelineExecutor(
    job_name="basic_filtering",
    pipeline=pipeline,
    tasks=2,
    time="00:05:00",
    partition="gpu",
    logging_dir="/tmp/logs/",
    slurm_logs_folder="/tmp/slurm_logs/",
    cpus_per_task=8,
    mem_per_cpu_gb=8,
)

executor.run()