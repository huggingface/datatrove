import pyarrow.parquet as pq

from datatrove.io import BaseInputDataFile, BaseInputDataFolder
from datatrove.pipeline.readers.base import BaseReader


class ParquetReader(BaseReader):
    name = "📒 Parquet"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        **kwargs,
    ):
        super().__init__(data_folder, **kwargs)

    def read_file(self, datafile: BaseInputDataFile):
        with datafile.open(binary=True) as f:
            with pq.ParquetFile(f) as pqf:
                for li, line in enumerate(pqf.iter_batches(batch_size=1)):
                    with self.stats.time_manager:
                        document = self.get_document_from_dict(line.to_pydict(), datafile, li)
                        if not document:
                            continue
                    yield document
