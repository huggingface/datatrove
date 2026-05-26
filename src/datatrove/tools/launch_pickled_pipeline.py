import argparse
import warnings

import dill

from datatrove.executor.base import PipelineExecutor
from datatrove.io import open_file


parser = argparse.ArgumentParser("Loads a pickled pipeline executor and launches it.")

parser.add_argument("path", type=str, help="Path to the pickled file (usually a file called executor.pik)")


def main():
    args = parser.parse_args()
    warnings.warn(
        "WARNING: Loading pickled files can execute arbitrary code. "
        "Only load files from trusted sources.",
        RuntimeWarning,
        stacklevel=2,
    )
    with open_file(args.path, "rb") as f:
        executor: PipelineExecutor = dill.load(f)
    executor.run()


if __name__ == "__main__":
    main()
