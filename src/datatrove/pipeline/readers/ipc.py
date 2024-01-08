from typing import Callable

import pyarrow as pa

from datatrove.io import BaseInputDataFile, BaseInputDataFolder
from datatrove.pipeline.readers.base import BaseReader


class IpcReader(BaseReader):
    name = "🪶 Ipc"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        limit: int = -1,
        stream: bool = False,
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
    ):
        super().__init__(data_folder, limit, progress, adapter, text_key, id_key, default_metadata)
        self.stream = stream
        # TODO: add option to disable reading metadata (https://github.com/apache/arrow/issues/13827 needs to be addressed first)

    @staticmethod
    def _iter_file_batches(datafile: BaseInputDataFile):
        with datafile.open(binary=True) as f:
            with pa.ipc.open_file(f) as ipc_reader:
                for i in range(ipc_reader.num_record_batches):
                    yield ipc_reader.get_batch(i)

    @staticmethod
    def _iter_stream_batches(datafile: BaseInputDataFile):
        with datafile.open(binary=True) as f:
            with pa.ipc.open_stream(f) as ipc_stream_reader:
                for batch in ipc_stream_reader:
                    yield batch

    def read_file(self, datafile: BaseInputDataFile):
        batch_iter = self._iter_file_batches(datafile) if not self.stream else self._iter_stream_batches(datafile)
        li = 0
        for batch in batch_iter:
            documents = []
            with self.track_time("batch"):
                for line in batch.to_pylist():
                    document = self.get_document_from_dict(line, datafile, li)
                    if not document:
                        continue
                    documents.append(document)
                    li += 1
            yield from documents
