from typing import Callable

import pyarrow.parquet as pq
from loguru import logger

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
        default_metadata: dict = None,
    ):
        super().__init__(data_folder, limit, progress, adapter, content_key, id_key, default_metadata)

    def read_file(self, datafile: BaseInputDataFile):
        with datafile.open(binary=True) as f:
            with pq.ParquetFile(f) as pqf:
                li = 0
                if self.content_key not in pqf.schema_arrow.names:
                    logger.warning(f"Input file {datafile.path} is without content, skipping.")
                for batch in pqf.iter_batches(batch_size=1000, columns=[self.content_key]):
                    with self.track_time():
                        for line in batch.to_pylist():
                            document = self.get_document_from_dict(line, datafile, li)
                            if not document:
                                continue
                            li += 1
                            yield document
