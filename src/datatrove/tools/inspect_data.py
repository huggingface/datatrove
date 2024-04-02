import argparse
import os.path
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from datatrove.io import DataFolder, get_datafolder
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import CSVReader, JsonlReader, ParquetReader, WarcReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils._import_utils import is_rich_available


"""
    Simple utility to visualize some tokenized text data directly using an appropriate reader.
"""

if not is_rich_available():
    raise ImportError("Please install `rich` to run this command (`pip install rich`).")


parser = argparse.ArgumentParser(
    "Manually inspect some RefinedWeb samples. "
    "Any unknown parameters will be passed to the reader (example: 'text_key=text')."
)

parser.add_argument(
    "path", type=str, nargs="?", help="Path to the data folder. Defaults to current directory.", default=os.getcwd()
)

parser.add_argument(
    "-r",
    "--reader",
    type=str,
    help="The type of Reader to use to read the data. "
    "By default it will be guessed from the file extension. "
    "Can be ('jsonl', 'parquet', 'csv' or 'warc')",
)

parser.add_argument(
    "-s", "--sample", type=float, help="Randomly sample a given % of samples. 1.0 to see all samples", default=1.0
)

parser.add_argument(
    "-l", "--label", type=str, help="Label the examples as good/bad and store at this location", default=""
)

console = Console()


def reader_class_from_name(reader_type):
    match reader_type:
        case "jsonl":
            return JsonlReader
        case "csv":
            return CSVReader
        case "parquet":
            return ParquetReader
        case "warc":
            return WarcReader
        case other:
            console.log(f"[red]Unknwon reader type {other}")
            sys.exit(-1)


def reader_factory(data_folder: DataFolder, reader_type: str = None, **kwargs):
    """

    Args:
      data_folder: DataFolder:
      reader_type: str:  (Default value = None)
      **kwargs:

    Returns:

    """
    data_files = data_folder.list_files()
    if not data_files:
        console.log(f'[red]Could not find any files in "{data_folder.path}"')
        sys.exit(-1)

    if not reader_type:
        match data_files[0][data_files[0].index(".") :]:
            case ".jsonl.gz" | ".jsonl" | ".json":
                reader_type = "jsonl"
            case ".csv":
                reader_type = "csv"
            case ".parquet":
                reader_type = "parquet"
            case ".warc.gz" | "arc.gz" | ".warc":
                reader_type = "warc"
            case other:
                console.log(f'[red]Could not find a matching reader for file extension "{other}"')
                sys.exit(-1)
    return reader_class_from_name(reader_type)(data_folder, **kwargs)


def get_filter_expr(text=None):
    """

    Args:
      text:  (Default value = None)

    Returns:

    """
    return (lambda x: eval(text)) if text else (lambda x: True)


def main():
    """ """
    args, extra_args = parser.parse_known_args()
    kwargs = dict(extra_arg.split("=") for extra_arg in extra_args)
    data_folder = get_datafolder(args.path)
    label_folder = get_datafolder(args.label) if args.label else None

    reader = reader_factory(data_folder, args.reader, **kwargs)

    sampler = SamplerFilter(args.sample)

    console.print(
        f'Loading samples from "{data_folder.path}" with {reader} and sampling_rate={args.sample}.\n'
        f"Samples are displayed full page one by one.\n"
        f"If you don't see any color you may run \"export PAGER='less -r'\"."
    )

    filter_expr_text = None
    if Confirm.ask(
        "Would you like to add a filtering expression? (ex: x.metadata['token_count'] > 5000)", default=False
    ):
        filter_expr_text = Confirm.get_input(console, "Type your filtering expression: ", password=False)
    filter_expr = get_filter_expr(filter_expr_text)

    good_samples = []
    bad_samples = []
    iterator = sampler(reader())
    try:
        for sample in iterator:
            if not filter_expr(sample):
                continue
            with console.pager(styles=True):
                console.print(
                    Panel(
                        f"[yellow]Data ID:[reset] {sample.id}\n"
                        f"[yellow]Metadata:[reset]\n"
                        + "\n".join(f"- [blue]{field}: [reset] {value}" for field, value in sample.metadata.items())
                    )
                )
                console.print(sample.text)
            if label_folder:
                result = Prompt.ask(
                    "To label as good/bad example enter 'g'/'b'. Enter 'q' to skip labelling and move to the next "
                    "sample. Enter 'e' (exit) to leave:",
                    console=console,
                    choices=["g", "b", "e", "q"],
                )
                if result == "g":
                    good_samples.append(sample)
                elif result == "b":
                    bad_samples.append(sample)
                elif result == "e":
                    break
    except Exception:
        console.print_exception()
    finally:
        if good_samples and label_folder:
            with JsonlWriter(label_folder, "good_samples.jsonl", compression=None) as writer:
                for sample in good_samples:
                    writer.write(sample)
        if bad_samples and label_folder:
            with JsonlWriter(label_folder, "bad_samples.jsonl", compression=None) as writer:
                for sample in bad_samples:
                    writer.write(sample)


if __name__ == "__main__":
    main()
