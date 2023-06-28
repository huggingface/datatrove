import gzip
import json

from datatrove.data import Document
from datatrove.pipeline.readers.base import BaseReader
from datatrove.io import InputDataFile


class JsonlReader(BaseReader):
    name = "🐿 Jsonl reader"

    def read_file(self, datafile: InputDataFile):
        with datafile.open(lambda x: gzip.open(x, 'rt')) as f:
            for line in f:
                with self.time_stats_manager:
                    try:
                        d = json.loads(line)
                        document = Document(**d)
                        document.metadata.setdefault('file_path', datafile.path)
                    except EOFError:
                        # logger.warning(f"EOFError reading path {path}")
                        continue

                yield document
