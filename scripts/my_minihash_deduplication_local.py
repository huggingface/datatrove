from datatrove.executor.local import LocalPipelineExecutor
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
from glob import glob
from loguru import logger

# 配置保持不变
minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
)

def process(input_folder, output_folder, n_job):
    INPUT_READER = JsonlReader(input_folder, text_key="text")

    # Stage 1: 计算MinHash签名
    stage1 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(output_folder=f"{output_folder}/signatures", config=minhash_config),
        ],
        tasks=n_job,
        workers=n_job,  # 如果本地环境支持并行执行，可以设置workers
        logging_dir=f"logs/signatures",
    )

    # Stage 2: 在每个桶中找到匹配的签名
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{output_folder}/signatures",
                output_folder=f"{output_folder}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        workers=minhash_config.num_buckets,
        logging_dir=f"logs/buckets",
        depends=stage1,
    )

    # Stage 3: 使用所有桶的结果创建重复项聚类
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{output_folder}/buckets",
                output_folder=f"{output_folder}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,
        logging_dir=f"logs/clusters",
        depends=stage2,
    )

    # Stage 4: 从原始输入数据中读取，并移除所有属于同一重复项聚类的数据，除了每个聚类中的一个样本
    stage4 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            TokensCounter(),  # 方便查看去重前后的标记数量变化
            MinhashDedupFilter(
                input_folder=f"{output_folder}/remove_ids",
                exclusion_writer=JsonlWriter(f"{output_folder}/removed"),
            ),
            JsonlWriter(output_folder=f"{output_folder}/deduplicated_output"),
        ],
        tasks=n_job,
        workers=n_job,
        logging_dir=f"logs/filter",
        depends=stage3,
    )

    print(stage4.run())

if __name__ == '__main__':
    input_folder = "/mnt/nas/xuruohao/fineweb/Accounting/"
    output_folder = "/mnt/nas/xuruohao/fineweb/Accounting-deduplicated/"
    
    # 使用 * 匹配所有文件
    json_files = glob(f"{input_folder}/*")
    njobs = len(json_files)
    logger.info(f"Contains {njobs} files, creating {njobs} jobs.")
    
    process(
        input_folder=input_folder,
        output_folder=output_folder,
        n_job=njobs,
    )