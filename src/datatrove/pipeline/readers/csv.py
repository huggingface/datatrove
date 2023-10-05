import csv
from typing import Callable, Literal

from loguru import logger

from datatrove.data import Document
from datatrove.io import BaseInputDataFolder, InputDataFile
from datatrove.pipeline.readers.base import BaseReader


class CSVReader(BaseReader):
    name = "🔢 CSV"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        content_column: str = "content",
        id_column: str = "data_id",
        compression: Literal["gzip", "zst"] | None = None,
        adapter: Callable = None,
        **kwargs,
    ):
        super().__init__(data_folder, **kwargs)
        self.content_column = content_column
        self.id_column = id_column
        self.compression = compression
        self.adapter = adapter if adapter else lambda d, path, li: self._base_adapter
        self.empty_warning = False

    def read_file(self, datafile: InputDataFile):
        with datafile.open(compression=self.compression) as f:
            csv_reader = csv.DictReader(f)
            for di, d in enumerate(csv_reader):
                with self.stats.time_manager:
                    if not d.get(self.content_column, None):
                        if not self.empty_warning:
                            self.empty_warning = True
                            logger.warning("Found document without content, skipping.")
                        continue
                    document = Document(**self.adapter(d, datafile.path, di))
                    document.metadata.setdefault("file_path", datafile.path)
                yield document

    def _base_adapter(self, d: dict, path: str, di: int):
        return {"content": d.pop(self.content_column), "data_id": d.pop(self.id_column, f"{path}_{di}"), "metadata": d}
