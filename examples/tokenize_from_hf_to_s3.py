import argparse
import os.path

from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="huggingface dataset name. Example: stas/openwebtext-10k")
parser.add_argument(
    "output_path", type=str, help="Where to save individual tokenization files and the final merged " "files."
)
parser.add_argument("-l", "--logs", type=str, help="path to logs folder", default="tokenization_logs")
parser.add_argument("-t", "--tokenizer", type=str, help="tokenizer to use", default="gpt2")
parser.add_argument(
    "--local",
    type=str,
    help="local working directory. You won't need it if output_path is a local "
    "path. /scratch/datatrove/tokenize/ by default.",
    default="/scratch/datatrove/tokenize/",
)
parser.add_argument(
    "-o",
    "--output_name",
    type=str,
    help="filename for the final output files. By default this will " "be `dataset-tokenizer`",
    default=None,
)
parser.add_argument("-s", "--split", type=str, help="dataset split. `train` by default", default="train")
parser.add_argument(
    "-tk",
    "--text_key",
    type=str,
    help="Column that actually contains the text to be " "tokenized. `text` by default.",
    default="text",
)
parser.add_argument(
    "-p", "--partition", type=str, help="Slurm partition to use. `hopper-prod` by default.", default="hopper-prod"
)
parser.add_argument("-ts", "--tasks", type=int, help="Number of tasks to run. 1000 by default", default=1000)

# this pipeline will pull the openwebtext dataset from the HF hub, locally tokenize it in parallel
# and then merge-shuffle everything into a final tokenized binary file
# you could also run this on slurm by using a SlurmPipelineExecutor

args = parser.parse_args()
DATASET_NAME = args.output_name  # name for the final dataset files
if not DATASET_NAME:
    DATASET_NAME = f"{args.dataset}-{args.tokenizer}".replace("/", "_")

LOGS_FOLDER = args.logs
WORKING_DIR = os.path.join(args.output_path, "tokenized-tasks")  # where to save the tokenized individual files
LOCAL_WORKING_DIR = args.local  # if your WORKING_DIR is a local path, you won't need this one
# and can set it to `None`
if LOCAL_WORKING_DIR:
    LOCAL_WORKING_DIR = os.path.join(LOCAL_WORKING_DIR, DATASET_NAME)
FINAL_OUTPUT_DIR = os.path.join(
    args.output_path, "merged-dataset"
)  # where to save the final merged tokenized dataset.
# This is the path you will pass to your training library so that it can read the tokenized data.


if __name__ == "__main__":
    # this first (distributed) step will tokenize and shuffle the files from each task
    dist_executor = SlurmPipelineExecutor(
        job_name=f"{DATASET_NAME}-tok1",
        pipeline=[
            HuggingFaceDatasetReader(
                dataset=args.dataset,  # dataset name
                dataset_options={
                    "split": args.split  # any other options that should be passed to load_dataset
                },
                text_key=args.text_key,  # the column that actually contains the text to be tokenized
            ),
            DocumentTokenizer(
                output_folder=WORKING_DIR,
                local_working_dir=LOCAL_WORKING_DIR,
                save_filename=f"{DATASET_NAME}_tokenized",
                tokenizer_name_or_path=args.tokenizer,
            ),
        ],
        # If you have a very small dataset, feel free to set this to "1" and remove the merge_executor
        tasks=args.tasks,  # or as many as you want.
        logging_dir=f"{LOGS_FOLDER}/tokenization",
        time="20:00:00",
        partition=args.partition,
    )

    # this final step randomly samples from the previously shuffled
    merge_executor = SlurmPipelineExecutor(
        job_name=f"{DATASET_NAME}-tok2",
        pipeline=[
            DocumentTokenizerMerger(
                input_folder=WORKING_DIR,
                output_folder=FINAL_OUTPUT_DIR,
                save_filename=f"{DATASET_NAME}",
            ),
        ],
        tasks=1,  # important
        logging_dir=f"{LOGS_FOLDER}/merged",
        time="50:00:00",
        partition=args.partition,
        depends=dist_executor,
    )

    merge_executor.run()
