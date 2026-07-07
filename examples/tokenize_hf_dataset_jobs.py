"""
Tokenize a Hugging Face dataset on Hugging Face Jobs, then merge-shuffle it into a single
tokenized dataset ready for a training library.

The `JobsPipelineExecutor` twin of `tokenize_from_hf_to_s3.py` (which uses Slurm). Two stages:
1. (fan-out) `HuggingFaceDatasetReader` -> `DocumentTokenizer`: each Job tokenizes + shuffles a shard.
2. (fan-in, `depends=` on stage 1) `DocumentTokenizerMerger`: merge-shuffle into the final files.

Same pipeline as the Slurm version; only the executor config differs (see `filter_hf_dataset_jobs.py`
for the full comparison). Two things specific to this multi-stage flow on Jobs:
- `output_path` MUST be a shared REMOTE path (hf:// or s3://) — both stages run on different machines
  and read/write the same `tokenized-tasks/` folder.
- The tokens pipeline needs the `processing` extra (there is no separate `tokens` extra): `tokenizers`
  and `regex` live there. So `dependencies=["datatrove[io,processing]"]`.
"""

import argparse
import os.path


parser = argparse.ArgumentParser("Tokenize an HF dataset on HF Jobs")
parser.add_argument("dataset", type=str, help="HF dataset name, e.g. stanfordnlp/imdb")
parser.add_argument(
    "output_path", type=str, help="REMOTE base path (e.g. hf://datasets/<you>/<name>) for tokenized + merged files"
)
parser.add_argument("-t", "--tokenizer", type=str, help="tokenizer to use", default="gpt2")
parser.add_argument("-s", "--split", type=str, help="dataset split", default="train")
parser.add_argument("-tk", "--text_key", type=str, help="text column", default="text")
parser.add_argument("-ts", "--tasks", type=int, help="number of tokenization tasks", default=100)
parser.add_argument("--workers", type=int, help="max concurrent Jobs (-1 = all at once)", default=20)
parser.add_argument("--flavor", type=str, help="HF Jobs hardware flavor", default="cpu-basic")

# tokens machinery lives in the `processing` extra (tokenizers + regex); `datasets` comes from `io`.
# Until JobsPipelineExecutor is released, install datatrove from your branch, e.g.:
#   "datatrove[io,processing] @ git+https://github.com/<user>/datatrove@<branch>"
DEPENDENCIES = ["datatrove[io,processing]"]

if __name__ == "__main__":
    args = parser.parse_args()
    from datatrove.executor import JobsPipelineExecutor
    from datatrove.pipeline.readers import HuggingFaceDatasetReader
    from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
    from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer

    name = f"{args.dataset}-{args.tokenizer}".replace("/", "_")
    working_dir = os.path.join(args.output_path, "tokenized-tasks")  # shared remote path — both stages read it
    final_dir = os.path.join(args.output_path, "merged-dataset")  # the path you pass to your training library

    tokenize = JobsPipelineExecutor(
        job_name=f"{name}-tok1",
        pipeline=[
            HuggingFaceDatasetReader(
                dataset=args.dataset,
                dataset_options={"split": args.split},
                text_key=args.text_key,
            ),
            DocumentTokenizer(
                output_folder=working_dir,
                local_working_dir="/tmp/datatrove-tokenize",  # Job-local scratch (ephemeral); staged then uploaded
                save_filename=f"{name}_tokenized",
                tokenizer_name_or_path=args.tokenizer,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=os.path.join(args.output_path, "logs", "tokenization"),
        flavor=args.flavor,
        dependencies=DEPENDENCIES,
        timeout="2h",
    )

    merge = JobsPipelineExecutor(
        job_name=f"{name}-tok2",
        pipeline=[DocumentTokenizerMerger(input_folder=working_dir, output_folder=final_dir, save_filename=name)],
        tasks=1,  # important: the merger is a single fan-in task
        logging_dir=os.path.join(args.output_path, "logs", "merged"),
        flavor=args.flavor,
        dependencies=DEPENDENCIES,
        timeout="2h",
        depends=tokenize,  # <-- launches + waits for all tokenization tasks, then merges
    )

    merge.run()
