import os
import logging
from glob import glob
from loguru import logger
import portalocker

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LanguageFilter, LambdaFilter
from datatrove.pipeline.writers.jsonl import JsonlWriter

# ===== 必须放在模块顶层，否则无法被远程 worker 访问 =====
PROCESSED_FILES_LOCK = "processed_files.lock"
PROCESSED_FILES_LOG = "processed_files.log"


def has_file_been_processed(filepath):
    if not os.path.exists(PROCESSED_FILES_LOG):
        return False
    with open(PROCESSED_FILES_LOG, "r") as f:
        processed = set(line.strip() for line in f)
    return filepath in processed


def mark_file_as_processed(filepath):
    with portalocker.Lock(PROCESSED_FILES_LOCK, mode="a", timeout=10) as f:
        f.write(f"{filepath}\n")


def delete_file_callback(path):
    """回调函数：文件处理完成后删除"""
    if has_file_been_processed(path):
        logger.info(f"File already processed by another worker: {path}")
        return

    try:
        os.remove(path)
        logger.info(f"[Callback] Deleted file: {path}")
        mark_file_as_processed(path)
    except Exception as e:
        logger.error(f"[Callback] Failed to delete file {path}: {e}")


# ===== 文件追踪器：用于检测何时完成处理一个文件 =====
class FileTracker:
    def __init__(self):
        self.current_source = None

    def track(self, document):
        source = getattr(document, "source", None)
        if source and source != self.current_source:
            if self.current_source:
                delete_file_callback(self.current_source)
            self.current_source = source
        return True  # 继续传递文档


# ===== 主流程函数 =====
def process(input_folder, output_folder, job_name, n_job, partition, file_name):
    os.makedirs(output_folder, exist_ok=True)
    logger.add(f"{output_folder}/processing_{file_name}.log", rotation="500 MB")
    logger.info(f"Processing {input_folder} with {n_job} jobs.")

    INPUT_READER = JsonlReader(
        input_folder,
        glob_pattern="*.tsv.gz",
        text_key="text"
    )

    MAIN_OUTPUT_PATH = output_folder
    SLURM_LOGS_FOLDER = f"/public/home/xuruohao/home/xuruohao/code/logs/r3_dclm/slurm_logs/lang-test/"
    LOGGING_DIR = f"/public/home/xuruohao/home/xuruohao/data/r3_dclm_logs/lang-test/"

    file_tracker = FileTracker()

    main_processing_executor = SlurmPipelineExecutor(
        job_name=f"{job_name}_filter",
        pipeline=[
            INPUT_READER,
            LambdaFilter(file_tracker.track),  # 使用 FileTracker 实例方法
            LanguageFilter(backend="glotlid", label_only=True),
            LambdaFilter(filter_function=lambda doc: doc.metadata["language"] in ['eng', 'eng_Latn']),
            JsonlWriter(MAIN_OUTPUT_PATH),
        ],
        tasks=n_job,
        time="300:00:00",
        logging_dir=f"{LOGGING_DIR}/{file_name}",
        slurm_logs_folder=f"{SLURM_LOGS_FOLDER}/{file_name}",  # must be local
        randomize_start_duration=180,
        mem_per_cpu_gb=12,
        partition=partition,
        qos="",
    )

    result = main_processing_executor.run()
    return result


# ===== 获取子目录并启动任务 =====
def get_subfolders(parent_folder):
    return [f.path for f in os.scandir(parent_folder) if f.is_dir()]


if __name__ == '__main__':
    input_folder_base = "/public/home/xuruohao/home/xuruohao/data/r3_dclm/"
    output_folder_base = "/public/home/xuruohao/home/xuruohao/data/r3_dclm_lang/"
    base_job_name = "lang"

    subfolders = ['/public/home/xuruohao/home/xuruohao/data/r3_dclm/subject=Mathematics2']

    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        input_folder = subfolder
        file_name = subfolder_name.replace("subject=", "")
        output_folder = os.path.join(output_folder_base, file_name)
        njobs = len(glob(f"{input_folder}/*.tsv.gz"))

        process(
            input_folder=input_folder,
            output_folder=output_folder,
            job_name=f"{base_job_name}_{file_name}",
            n_job=njobs,
            partition="operation",
            file_name=file_name
        )