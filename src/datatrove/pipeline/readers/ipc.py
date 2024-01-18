from typing import Callable

import pyarrow as pa

from datatrove.io import DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


class IpcReader(BaseDiskReader):
    name = "🪶 Ipc"

    def __init__(
        self,
        data_folder: DataFolderLike,
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

    def _iter_file_batches(self, filepath: str):
        with self.data_folder.open(filepath, "rb") as f:
            with pa.ipc.open_file(f) as ipc_reader:
                for i in range(ipc_reader.num_record_batches):
                    yield ipc_reader.get_batch(i)

    def _iter_stream_batches(self, filepath: str):
        with self.data_folder.open(filepath, "rb") as f:
            with pa.ipc.open_stream(f) as ipc_stream_reader:
                for batch in ipc_stream_reader:
                    yield batch

    def read_file(self, filepath: str):
        batch_iter = self._iter_file_batches(filepath) if not self.stream else self._iter_stream_batches(filepath)
        li = 0
        for batch in batch_iter:
            documents = []
            with self.track_time("batch"):
                for line in batch.to_pylist():
                    document = self.get_document_from_dict(line, filepath, li)
                    if not document:
                        continue
                    documents.append(document)
                    li += 1
            yield from documents
