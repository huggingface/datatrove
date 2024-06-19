from typing import Callable

from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


class IpcReader(BaseDiskReader):
    """Read data from Apache Arrow IPC files.

    Args:
        data_folder: a str, tuple or DataFolder object representing a path/filesystem
        paths_file: optionally provide a file with one path per line (without the `data_folder` prefix) to read.
        limit: limit the number of documents to read. Useful for debugging
        skip: skip the first n rows
        stream: if True, will read the file as a stream (default: False)
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Takes as input: (self, data: dict, path: str, id_in_file: int | str)
                self allows access to self.text_key and self.id_key
            Returns: a dict with at least a "text" and "id" keys
        text_key: the key containing the text data (default: "text").
        id_key: the key containing the id for each sample (default: "id").
        default_metadata: a dictionary with any data that should be added to all samples' metadata
        recursive: whether to search files recursively. Ignored if paths_file is provided
        glob_pattern: pattern that all files must match exactly to be included (relative to data_folder). Ignored if paths_file is provided
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use with dedup blocks
    """

    name = "🪶 Ipc"
    _requires_dependencies = ["pyarrow"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
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
            paths_file,
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
