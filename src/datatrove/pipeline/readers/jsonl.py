import json
from typing import Callable

from datatrove.data import Document
from datatrove.io import BaseInputDataFolder, InputDataFile
from datatrove.pipeline.readers.base import BaseReader


class JsonlReader(BaseReader):
    name = "🐿 Jsonl"

    def __init__(self, data_folder: BaseInputDataFolder, gzip: bool = True, adapter: Callable = None, **kwargs):
        super().__init__(data_folder, **kwargs)
        self.gzip = gzip
        self.adapter = adapter if adapter else lambda d, path, li: d

    def read_file(self, datafile: InputDataFile):
        with datafile.open(gzip=self.gzip) as f:
            for li, line in enumerate(f):
                with self.stats.time_manager:
                    try:
                        d = json.loads(line)
                        document = Document(**self.adapter(d, datafile.path, li))
                        document.metadata.setdefault("file_path", datafile.path)
                    except EOFError:
                        # logger.warning(f"EOFError reading path {path}")
                        continue
                yield document
