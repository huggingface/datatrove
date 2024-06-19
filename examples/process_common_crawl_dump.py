import sys

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


# DUMP should be given as an argument. Example: CC-MAIN-2023-23
if len(sys.argv) != 2:
    print("Argument required: dump name")
    sys.exit(-1)
DUMP = sys.argv[1]

MAIN_OUTPUT_PATH = "s3://some_s3_bucket/base_processing/"

executor = SlurmPipelineExecutor(
    job_name=f"cc_{DUMP}",
    pipeline=[
        WarcReader(
            f"s3://commoncrawl/crawl-data/{DUMP}/segments/",
            glob_pattern="*/warc/*",  # we want the warc files
            default_metadata={"dump": DUMP},
        ),
        URLFilter(exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/url/{DUMP}")),
        Trafilatura(favour_precision=True),
        LanguageFilter(
            exclusion_writer=JsonlWriter(
                f"{MAIN_OUTPUT_PATH}/non_english/",
                output_filename="${language}/" + DUMP + "/${rank}.jsonl.gz",  # folder structure: language/dump/file
            )
        ),
        GopherRepetitionFilter(exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/repetitive/{DUMP}")),
        GopherQualityFilter(exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/quality/{DUMP}")),
        JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DUMP}"),
    ],
    tasks=8000,
    time="10:00:00",
    logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{DUMP}",
    slurm_logs_folder=f"logs/process_dump/processing/base_processing/{DUMP}/slurm_logs",
    randomize_start_duration=180,
    mem_per_cpu_gb=2,
    partition="hopper-cpu",
)
executor.run()
