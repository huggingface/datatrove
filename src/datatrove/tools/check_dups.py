from datatrove.pipeline.readers import CSVReader, JsonlReader, ParquetReader, WarcReader
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.counter import TokensCounter
from datatrove.pipeline.filters.language_filter import LanguageFilter


class DuplicatesChecker(PipelineStep):
    def __init__(self, field: str):
        super().__init__()
        self.field = field
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        found_fields = set()
        dups = 0
        for sample in data:
            field = getattr(sample, self.field, None)
            if field in found_fields:
                print(f"Found duplicate {self.field} in {sample.id}")
                dups += 1
            else:
                found_fields.add(field)
            print(sample.metadata)
            yield sample

        print(f"Found {dups} duplicates of {self.field}")


def main():
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--input_folder", type=str, required=True)
    args = args_parser.parse_args()
    LocalPipelineExecutor(
        [
            JsonlReader(args.input_folder, glob_pattern="*.jsonl.gz", doc_progress=True),
            DuplicatesChecker(field="text"),
        ]
    ).run()

if __name__ == "__main__":
    main()
