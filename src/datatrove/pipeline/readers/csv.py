import csv
from typing import Callable, Literal

from datatrove.io import DataFolderLike
from datatrove.pipeline.readers.base import BaseReader


class CsvReader(BaseReader):
    name = "🔢 Csv"

    def __init__(
        self,
        data_folder: DataFolderLike,
        compression: Literal["guess", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
    ):
        super().__init__(
            data_folder, limit, progress, adapter, text_key, id_key, default_metadata, recursive, glob_pattern
        )
        self.compression = compression
        self.empty_warning = False

    def read_file(self, filepath: str):
        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            csv_reader = csv.DictReader(f)
            for di, d in enumerate(csv_reader):
                with self.track_time():
                    document = self.get_document_from_dict(d, filepath, di)
                    if not document:
                        continue
                yield document


CSVReader = CsvReader
