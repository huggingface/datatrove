import os
from glob import glob
from loguru import logger

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import (
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
    C4QualityFilter,
    LambdaFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter



def process(input_folder, output_folder, job_name, n_job, partition, file_name):
    # 确保输出文件夹存在
    import json
    os.makedirs(output_folder, exist_ok=True)
    
    logger.info(f"Processing {input_folder} with {n_job} jobs.")
    
    INPUT_READER = JsonlReader(input_folder, glob_pattern="*.jsonl.gz", text_key="text")
    MAIN_OUTPUT_PATH = output_folder
    SLURM_LOGS_FOLDER = f"/public/home/xuruohao/home/xuruohao/code/logs/r3_dclm/slurm_logs/quality_no_c4/"
    LOGGING_FOLDER = f"/public/home/xuruohao/home/xuruohao/data/r3_dclm_logs/quality_no_c4/"

    main_processing_executor = SlurmPipelineExecutor(
        job_name=f"{job_name}",
        pipeline=[
            INPUT_READER,
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(os.path.join(MAIN_OUTPUT_PATH,'remove',file_name,'1_GopherRepetition'))
            ),
            GopherQualityFilter(
                exclusion_writer=JsonlWriter(os.path.join(MAIN_OUTPUT_PATH,'remove',file_name,'2_GopherQuality'))
            ),
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(os.path.join(MAIN_OUTPUT_PATH,'remove',file_name,'3_FineWebQuality'))
            ),
            JsonlWriter(os.path.join(MAIN_OUTPUT_PATH,'result',file_name)),
        ],
        tasks=n_job,
        time="300:00:00",
        logging_dir=os.path.join(LOGGING_FOLDER,file_name),
        slurm_logs_folder=os.path.join(SLURM_LOGS_FOLDER,file_name),  # must be local
        randomize_start_duration=180,  # don't hit the bucket all at once with the list requests
        mem_per_cpu_gb=12,
        partition=partition,
        qos="",
    )
    print(main_processing_executor.run())


def get_subfolders(parent_folder):
    """获取指定目录下的所有子文件夹"""
    return [f.path for f in os.scandir(parent_folder) if f.is_dir()]

if __name__ == '__main__':
    input_folder_base = "/public/home/xuruohao/home/xuruohao/data/r3_dclm_lang/"
    output_folder_base = "/public/home/xuruohao/home/xuruohao/data/r3_dclm_quality_no_c4/"
    base_job_name = "quality_2"

    # 获取该文件夹下的所有子文件夹
    subfolders = get_subfolders(input_folder_base)
    # subfolders = ['/public/home/xuruohao/home/xuruohao/data/r3_dclm_lang/Mathematics']
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        input_folder = subfolder
        file_name = subfolder_name
        output_folder = os.path.join(output_folder_base, file_name)
        njobs = len(glob(f"{input_folder}/*.jsonl.gz"))

        # 调用process函数处理每个子文件夹
        process(
            input_folder=input_folder,
            output_folder=output_folder_base,
            job_name=f"{base_job_name}_{file_name}",
            n_job=njobs,
            partition="operation",
            file_name=file_name
        )
