from datatrove.executor.base import PipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.io import S3InputDataFolder, S3OutputDataFolder
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger


pipeline = [
    DocumentTokenizerMerger(
        input_folder=S3InputDataFolder(
            path="s3://extreme-scale-datasets/refined-web/tokenized/",
            local_path="/scratch/guilherme/refined-web/tokenized/",
        ),
        output_folder=S3OutputDataFolder(
            path="s3://extreme-scale-datasets/refined-web/standard/",
            local_path="/scratch/guilherme/refined-web/merged/",
        ),
        save_filename="refined_web",
    ),
]

executor: PipelineExecutor = SlurmPipelineExecutor(
    pipeline=pipeline,
    tasks=1,
    time="50:00:00",
    partition="production-cluster",
    logging_dir="/fsx/guilherme/logs/tokenize_refined_web_merged",
    mem_per_cpu_gb=11,
)
executor.run()
