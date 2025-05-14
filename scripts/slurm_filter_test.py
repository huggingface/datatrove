import os
from glob import glob
from loguru import logger

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import (
    GopherRepetitionFilter,
    GopherQualityFilter,
    C4QualityFilter,
    FineWebQualityFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


def process_filter(input_folder, output_folder, job_name, n_job, partition, file_name, filter_type):
    """
    处理单个过滤器任务。
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param job_name: 任务名称
    :param n_job: 并行任务数
    :param partition: SLURM 分区
    :param file_name: 文件名标识
    :param filter_type: 过滤器类型
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    logger.info(f"Processing {input_folder} with filter {filter_type}.")

    # 定义输入读取器
    INPUT_READER = JsonlReader(input_folder, glob_pattern="*.jsonl", text_key="text")
    FILTERING_OUTPUT_PATH = f"{output_folder}/{filter_type}"

    SLURM_LOGS_FOLDER = f"/public/home/xuruohao/home/xuruohao/code/logs/subject_exp/slurm_logs/"
    LOGGING_FOLDER  = f"/public/home/xuruohao/home/xuruohao/data/logs/subject_exp/"

    # 根据过滤器类型选择过滤器
    if filter_type == "gopher_rep":
        filter_task = GopherRepetitionFilter(
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/")
        )
    elif filter_type == "gopher_qual":
        filter_task = GopherQualityFilter(
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/")
        )
    elif filter_type == "c4":
        filter_task = C4QualityFilter(
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/")
        )
    elif filter_type == "fineweb_qual":
        filter_task = FineWebQualityFilter(
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/")
        )
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # 创建 SLURM 执行器
    executor = SlurmPipelineExecutor(
        job_name=f"{job_name}_{filter_type}",
        pipeline=[
            INPUT_READER,
            filter_task,
            JsonlWriter(f"{FILTERING_OUTPUT_PATH}/output/"),
        ],
        tasks=n_job,
        time="300:00:00",
        logging_dir=f"{LOGGING_FOLDER}/{filter_type}/{file_name}",
        slurm_logs_folder=f"{SLURM_LOGS_FOLDER}/{filter_type}/{file_name}",
        randomize_start_duration=180,  # 避免同时启动所有任务
        mem_per_cpu_gb=12,
        partition=partition,
        qos="",
    )

    # 启动任务
    executor.run()


def get_subfolders(parent_folder):
    """获取指定目录下的所有子文件夹"""
    return [f.path for f in os.scandir(parent_folder) if f.is_dir()]


if __name__ == '__main__':
    input_folder_base = "/public/home/xuruohao/home/xuruohao/data/exp/"
    output_folder_base = "/public/home/xuruohao/home/xuruohao/data/exp_output/"
    base_job_name = "exp"

    # 获取该文件夹下的所有子文件夹
    subfolders = get_subfolders(input_folder_base)

    # 定义过滤器类型列表
    filter_types = ["gopher_rep", "gopher_qual", "c4", "fineweb_qual"]

    for subfolder in subfolders:
        file_name = os.path.basename(subfolder)
        input_folder = subfolder
        output_folder = os.path.join(output_folder_base, file_name)

        njobs = len(glob(f"{input_folder}/*.jsonl"))

        # 对每个子文件夹分别运行 4 个过滤器
        for filter_type in filter_types:
            process_filter(
                input_folder=input_folder,
                output_folder=output_folder,
                job_name=f"{base_job_name}_{file_name}",
                n_job=njobs,
                partition="operation",
                file_name=file_name,
                filter_type=filter_type,
            )
