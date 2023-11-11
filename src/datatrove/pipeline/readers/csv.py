import csv
from typing import Literal

from datatrove.io import BaseInputDataFile, BaseInputDataFolder
from datatrove.pipeline.readers.base import BaseReader


class CSVReader(BaseReader):
    name = "🔢 CSV"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        compression: Literal["guess", "gzip", "zst"] | None = "guess",
        **kwargs,
    ):
        super().__init__(data_folder, **kwargs)
        self.compression = compression
        self.empty_warning = False

    def read_file(self, datafile: BaseInputDataFile):
        with datafile.open(compression=self.compression) as f:
            csv_reader = csv.DictReader(f)
            for di, d in enumerate(csv_reader):
                with self.track_time():
                    document = self.get_document_from_dict(d, datafile, di)
                    if not document:
                        continue
                yield document
