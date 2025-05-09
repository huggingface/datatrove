import os

from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.tokens.megatron_tokenizer import MegatronTokenizer
from datatrove.pipeline.tokens.megatron_merger import MegatronTokenizerMerger
from datatrove.pipeline.tokens.distributed_megatron_merger import DistTokenizerMergerPlanner, DistTokenizerMergerExecutor

def process(input_folder, output_folder, filename, job_name, tokenizer, n_job):
    merge = True
    pipeline_1 = [
        JsonlReader(
            input_folder,
            glob_pattern="*.jsonl.gz",
            text_key="text",
        ),
        MegatronTokenizer(
            output_folder=f"{output_folder}",
            local_working_dir=f"{output_folder}/scratch/",
            save_filename=f"{filename}",
            tokenizer_name_or_path=f"{tokenizer}",
            batch_size=8,
            shuffle=False,
        ),
    ]

    SLURM_LOGS_FOLDER = f"/public/home/xuruohao/home/xuruohao/code/logs/r3_dclm/slurm_logs/token/"
    LOGGING_DIR = f"/public/home/xuruohao/home/xuruohao/data/r3_dclm_logs/token/"

    executor_1: PipelineExecutor = SlurmPipelineExecutor(
        pipeline=pipeline_1,
        cpus_per_task=1,
        mem_per_cpu_gb=16,
        job_name=f"{job_name}",
        partition="normal",
        time="12:00:00",
        logging_dir=f"{LOGGING_DIR}/{filename}/",
        slurm_logs_folder=f"{SLURM_LOGS_FOLDER}/{filename}",
        workers=n_job,
        tasks=n_job,
        qos=""
    )

    print(executor_1.run())


def get_subfolders(parent_folder):
    """获取指定目录下的所有子文件夹"""
    return [f.path for f in os.scandir(parent_folder) if f.is_dir()]

def get_folder_size(folder_path):
    """计算文件夹的总大小（包括子文件夹和文件）。"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            # 跳过符号链接等非普通文件
            if os.path.isfile(file_path) and not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

def calculate_n_job(folder_size):
    """根据文件夹大小计算 n_job，每 4GB 设置一个 n_job。"""
    # 将字节转换为 GB，并向上取整
    size_in_gb = folder_size / (1024 ** 3)
    n_job = int(size_in_gb // 4) + (1 if size_in_gb % 4 > 0 else 0)
    return n_job


if __name__ == "__main__":
    input_folder_base = "/public/home/xuruohao/home/xuruohao/data/r3_dclm_minhash/result/"
    output_folder_base = "/public/home/xuruohao/home/xuruohao/data/r3_dclm_tokens/"
    base_job_name = "token"
    
    # 获取该文件夹下的所有子文件夹
    # subfolders = get_subfolders(input_folder_base)
    subfolders = ['/public/home/xuruohao/home/xuruohao/data/r3_dclm_minhash/result/Mathematics']
    
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        
        input_folder = subfolder
        file_name = subfolder_name
        output_folder = os.path.join(output_folder_base, file_name)
            
        # 计算文件夹大小并设置 n_job
        folder_size = get_folder_size(input_folder)
        n_job = calculate_n_job(folder_size)
            
        # 调用 process 函数处理每个子文件夹
        process(
            input_folder=input_folder,
            output_folder=output_folder,
            filename=file_name,
            job_name=f"{base_job_name}_{file_name}",
            tokenizer="/public/home/xuruohao/home/xuruohao/zjllm-llama3-tokenizer/",
            n_job=n_job,
        )
            