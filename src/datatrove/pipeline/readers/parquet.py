from typing import Callable

import pyarrow.parquet as pq

from datatrove.io import BaseInputDataFile, BaseInputDataFolder
from datatrove.pipeline.readers.base import BaseReader


class ParquetReader(BaseReader):
    name = "📒 Parquet"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        content_key: str = "content",
        id_key: str = "data_id",
    ):
        super().__init__(data_folder, limit, progress, adapter, content_key, id_key)

    def read_file(self, datafile: BaseInputDataFile):
        with datafile.open(binary=True) as f:
            with pq.ParquetFile(f) as pqf:
                for li, line in enumerate(pqf.iter_batches(batch_size=1)):
                    with self.track_time():
                        document = self.get_document_from_dict(line.to_pydict(), datafile, li)
                        if not document:
                            continue
                    yield document
