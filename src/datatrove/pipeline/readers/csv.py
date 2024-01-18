import csv
from typing import Callable, Literal

from datatrove.io import DataFolderLike
from datatrove.pipeline.readers.base import BaseReader


class CSVReader(BaseReader):
    name = "🔢 CSV"

    def __init__(
        self,
        data_folder: DataFolderLike,
        compression: Literal["guess", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        content_key: str = "content",
        id_key: str = "data_id",
        default_metadata: dict = None,
    ):
        super().__init__(data_folder, limit, progress, adapter, content_key, id_key, default_metadata)
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
