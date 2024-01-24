from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


# this pipeline will pull the openwebtext dataset from the HF hub, locally tokenize it in parallel
# and then merge-shuffle everything into a final tokenized binary file
# you could also run this on slurm by using a SlurmPipelineExecutor


LOGS_FOLDER = "my_tokenization_logs"
WORKING_DIR = "s3://my-s3-bucket/openwebtext/tokenized-tasks/"  # where to save the tokenized individual files
LOCAL_WORKING_DIR = "/scratch/my-local-working-dir"  # if your WORKING_DIR is a local path, you won't need this one
# and can set it to `None`
FINAL_OUTPUT_DIR = "s3://my-s3-bucket/openwebtext/dataset/"  # where to save the final merged tokenized dataset.
# This is the path you will pass to your training library so that it can read the tokenized data.

DATASET_NAME = "openwebtext"  # name for the final dataset files

# this first (distributed) step will tokenize and shuffle the files from each task
dist_executor = LocalPipelineExecutor(
    pipeline=[
        HuggingFaceDatasetReader(
            dataset="stas/openwebtext-10k",  # dataset name
            dataset_options={
                "split": "train"  # any other options that should be passed to load_dataset
            },
            text_key="text",  # the column that actually contains the text to be tokenized
        ),
        DocumentTokenizer(
            output_folder=WORKING_DIR,
            local_working_dir=LOCAL_WORKING_DIR,
            save_filename=f"{DATASET_NAME}_tokenized",
        ),
    ],
    # If you have a very small dataset, feel free to set this to "1" and remove the merge_executor
    tasks=1000,  # or as many as you want.
    workers=16,  # set this to the number of cpus that you have available
    logging_dir=f"{LOGS_FOLDER}/tokenization",
)

# this final step randomly samples from the previously shuffled
merge_executor = LocalPipelineExecutor(
    pipeline=[
        DocumentTokenizerMerger(
            input_folder=WORKING_DIR,
            output_folder=FINAL_OUTPUT_DIR,
            save_filename=f"{DATASET_NAME}",
        ),
    ],
    tasks=1,  # important
    logging_dir=f"{LOGS_FOLDER}/merged",
)


if __name__ == "__main__":
    dist_executor.run()
    merge_executor.run()
