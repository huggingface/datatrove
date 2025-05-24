import os
from glob import glob
from loguru import logger

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    HashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter


def process(input_folder, output_folder, job_name, n_job, partition, file_name):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    logger.info(f"Processing {input_folder} with {n_job} jobs.")

    # 配置MinHash参数
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            precision=64
        ),
        num_buckets=14,
        hashes_per_bucket=8,
    )
    
    INPUT_READER = JsonlReader(input_folder, glob_pattern="*.jsonl.gz", text_key="text")

    MINHASH_BASE_PATH = os.path.join(output_folder,"tmp",file_name)

    LOGS_FOLDER = f"/public/home/xuruohao/home/xuruohao/data/r3_dclm_logs/minhash/"
    LOCAL_LOGS_FOLDER = f"/public/home/xuruohao/home/xuruohao/code/logs/r3_dclm/slurm_logs/minhash/"

    # Stage 1: 计算MinHash签名
    stage1 = SlurmPipelineExecutor(
        job_name=f"{job_name}_stage1",
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f"{MINHASH_BASE_PATH}/signatures", config=minhash_config
            ),
        ],
        tasks=n_job,
        workers=n_job,
        time="10:00:00",
        partition=partition,
        logging_dir=f"{LOGS_FOLDER}/{file_name}/signatures",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/{file_name}/signatures",
        mem_per_cpu_gb=12,
        qos=""
    )

    # Stage 2: 在每个桶中查找匹配项
    stage2 = SlurmPipelineExecutor(
        job_name=f"{job_name}_stage2",
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{MINHASH_BASE_PATH}/signatures",
                output_folder=f"{MINHASH_BASE_PATH}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        workers=minhash_config.num_buckets,
        time="90:00:00",
        partition=partition,
        logging_dir=f"{LOGS_FOLDER}/{file_name}/buckets",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/{file_name}/buckets",
        mem_per_cpu_gb=128,
        depends=stage1,
        qos=""
    )

    # Stage 3: 创建重复项集群
    stage3 = SlurmPipelineExecutor(
        job_name=f"{job_name}_stage3",
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{MINHASH_BASE_PATH}/buckets",
                output_folder=f"{MINHASH_BASE_PATH}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,
        time="90:00:00",
        partition=partition,
        logging_dir=f"{LOGS_FOLDER}/{file_name}/clusters",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/{file_name}/clusters",
        mem_per_cpu_gb=70,
        cpus_per_task=2,
        depends=stage2,
        qos=""
    )

    # Stage 4: 过滤掉重复项并输出去重后的数据
    stage4 = SlurmPipelineExecutor(
    job_name=f"{job_name}_stage4",
    pipeline=[
        INPUT_READER,
        #TokensCounter("/public/home/xuruohao/home/xuruohao/code/tokenizer.json"),  # 统计token数量变化情况
        MinhashDedupFilter(
            input_folder=f"{MINHASH_BASE_PATH}/remove_ids",
            exclusion_writer=JsonlWriter(os.path.join(output_folder, "removed",file_name)),
        ),
        JsonlWriter(os.path.join(output_folder,'result',file_name)),
    ],
    tasks=n_job,
    workers=n_job,
    time="50:00:00",
    partition=partition,
    mem_per_cpu_gb=32,
    logging_dir=f"{LOGS_FOLDER}/{file_name}/filter",
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/{file_name}/filter",
    depends=stage3,
    qos=""
    )

    # 执行并打印结果
    result = stage4.run()
    print(result)


def get_subfolders(parent_folder):
    """获取指定目录下的所有子文件夹"""
    return [f.path for f in os.scandir(parent_folder) if f.is_dir()]

if __name__ == '__main__':
    input_folder_base = "/public/home/xuruohao/home/xuruohao/data/r3_dclm_quality/result/"
    output_folder_base = "/public/home/xuruohao/home/xuruohao/data/r3_dclm_minhash/"
    base_job_name = "dedupe"

    # 获取该文件夹下的所有子文件夹
    # subfolders = get_subfolders(input_folder_base)
    subfolders = ['/public/home/xuruohao/home/xuruohao/data/r3_dclm_quality/result/Mathematics']
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