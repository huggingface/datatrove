from typing import Callable

import pyarrow.parquet as pq

from datatrove.data import Document
from datatrove.io import BaseInputDataFolder, InputDataFile
from datatrove.pipeline.readers.base import BaseReader


class ParquetReader(BaseReader):
    name = "📒 Parquet"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        adapter: Callable = None,
        **kwargs,
    ):
        super().__init__(data_folder, **kwargs)
        self.adapter = adapter if adapter else lambda d, path, li: d

    def read_file(self, datafile: InputDataFile):
        with datafile.open(binary=True) as f:
            with pq.ParquetFile(f) as pqf:
                for li, line in enumerate(pqf.iter_batches(batch_size=1)):
                    with self.stats.time_manager:
                        document = Document(**self.adapter(line.to_pydict(), datafile.path, li))
                        document.metadata.setdefault("file_path", datafile.path)
                    yield document
