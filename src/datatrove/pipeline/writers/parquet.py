from collections import defaultdict
from typing import IO, Callable

from loguru import logger

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike
from datatrove.pipeline.writers.disk_base import DiskWriter


class ParquetWriter(DiskWriter):
    default_output_filename: str = "${rank}.parquet"
    name = "📒 Parquet"
    _requires_dependencies = ["pyarrow"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: str | None = None,
        adapter: Callable = None,
        batch_size: int = 1000,
    ):
        super().__init__(output_folder, output_filename, compression, adapter, mode="wb")
        self._writers = {}
        self._batches = defaultdict(list)
        self.batch_size = batch_size

    def _write_batch(self, filename):
        if not self._batches[filename]:
            return
        import pyarrow as pa

        # prepare batch
        batch = pa.RecordBatch.from_pylist(self._batches.pop(filename))
        # write batch
        self._writers[filename].write_batch(batch)

    def _write(self, document: dict, file_handler: IO, filename: str):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if filename not in self._writers:
            self._writers[filename] = pq.ParquetWriter(
                file_handler, schema=pa.RecordBatch.from_pylist([document]).schema
            )
        self._batches[filename].append(document)
        if len(self._batches[filename]) == self.batch_size:
            self._write_batch(filename)

    def close(self):
        logger.info("CLOSE ON PW")
        for filename in list(self._batches.keys()):
            self._write_batch(filename)
        for writer in self._writers.values():
            writer.close()
        self._batches.clear()
        self._writers.clear()
        super().close()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        logger.info("RUN ON pw")
        super().run(data, rank, world_size)
