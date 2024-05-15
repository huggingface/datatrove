from typing import Callable

from datatrove.io import DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


class IpcReader(BaseDiskReader):
    """Read data from Apache Arrow IPC files.

    Args:
        data_folder: the data folder to read from
        limit: limit the number of IPC documents to read
        skip: skip the first n rows
        stream: if True, will read the file as a stream (default: False)
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Take as input: data: dict, path: str, id_in_file: int | str
            Return: a dict with at least a "text" key
        text_key: key to use for the text in the default adapter (default: "text"). Ignored if you provide your own `adapter`
        id_key: key to use for the id in the default adapter (default: "id"). Ignored if you provide your own `adapter`
        default_metadata: default metadata to add to all documents
        recursive: if True, will read files recursively in subfolders (default: True)
        glob_pattern: a glob pattern to filter files to read (default: None)
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use
            with dedup blocks
    """

    name = "🪶 Ipc"
    _requires_dependencies = ["pyarrow"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        limit: int = -1,
        skip: int = 0,
        stream: bool = False,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )
        self.stream = stream
        # TODO: add option to disable reading metadata (https://github.com/apache/arrow/issues/13827 needs to be addressed first)

    def _iter_file_batches(self, filepath: str):
        import pyarrow as pa

        with self.data_folder.open(filepath, "rb") as f:
            with pa.ipc.open_file(f) as ipc_reader:
                for i in range(ipc_reader.num_record_batches):
                    yield ipc_reader.get_batch(i)

    def _iter_stream_batches(self, filepath: str):
        import pyarrow as pa

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
