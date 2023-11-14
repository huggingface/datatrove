import argparse

import dill

from datatrove.executor.base import PipelineExecutor
from datatrove.io import BaseInputDataFile


parser = argparse.ArgumentParser("Loads a pickled pipeline executor and launches it.")

parser.add_argument("path", type=str, help="Path to the pickled file (usually a file called executor.pik)")


def main():
    args = parser.parse_args()
    executor_path = BaseInputDataFile.from_path(args.path)
    with executor_path.open_binary() as f:
        executor: PipelineExecutor = dill.load(f)
    executor.run()


if __name__ == "__main__":
    main()
