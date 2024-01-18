from typing import Callable

import pyarrow.parquet as pq

from datatrove.io import DataFolderLike
from datatrove.pipeline.readers.base import BaseReader


class ParquetReader(BaseReader):
    name = "📒 Parquet"

    def __init__(
        self,
        data_folder: DataFolderLike,
        limit: int = -1,
        batch_size: int = 1000,
        read_metadata: bool = True,
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
    ):
        super().__init__(data_folder, limit, progress, adapter, text_key, id_key, default_metadata)
        self.batch_size = batch_size
        self.read_metadata = read_metadata

    def read_file(self, filepath: str):
        with self.data_folder.open(filepath, "rb") as f:
            with pq.ParquetFile(f) as pqf:
                li = 0
                columns = [self.text_key, self.id_key] if not self.read_metadata else None
                for batch in pqf.iter_batches(batch_size=self.batch_size, columns=columns):
                    documents = []
                    with self.track_time("batch"):
                        for line in batch.to_pylist():
                            document = self.get_document_from_dict(line, filepath, li)
                            if not document:
                                continue
                            documents.append(document)
                            li += 1
                    yield from documents
