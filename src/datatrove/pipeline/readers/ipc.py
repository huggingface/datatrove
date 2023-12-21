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
        progress: bool = False,
        adapter: Callable = None,
        content_key: str = "content",
        id_key: str = "data_id",
        default_metadata: dict = None,
    ):
        super().__init__(data_folder, limit, progress, adapter, content_key, id_key, default_metadata)
        # TODO: add option to disable reading metadata (https://github.com/apache/arrow/issues/13827 needs to be addressed first)

    def read_file(self, datafile: BaseInputDataFile):
        with datafile.open(binary=True) as f:
            with pa.ipc.open_file(f) as ipc_reader:
                li = 0
                for i in range(ipc_reader.num_record_batches):
                    batch = ipc_reader.get_batch(i)
                    documents = []
                    with self.track_time("batch"):
                        for line in batch.to_pylist():
                            document = self.get_document_from_dict(line, datafile, li)
                            if not document:
                                continue
                            documents.append(document)
                            li += 1
                    yield from documents
