from typing import Callable

from datatrove.io import DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


class ParquetReader(BaseDiskReader):
    """Read data from Parquet files.
        Will read each batch as a separate document.

    Args:
        data_folder: the data folder to read from
        limit: limit the number of Parquet rows to read
        skip: skip the first n rows
        batch_size: the batch size to use (default: 1000)
        read_metadata: if True, will read the metadata (default: True)
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

    name = "📒 Parquet"
    _requires_dependencies = ["pyarrow"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        limit: int = -1,
        skip: int = 0,
        batch_size: int = 1000,
        read_metadata: bool = True,
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
        self.batch_size = batch_size
        self.read_metadata = read_metadata

    def read_file(self, filepath: str):
        import pyarrow.parquet as pq

        with self.data_folder.open(filepath, "rb") as f:
            with pq.ParquetFile(f) as pqf:
                li = 0
                columns = [self.text_key, self.id_key] if not self.read_metadata else None
                for batch in pqf.iter_batches(batch_size=self.batch_size, columns=columns):
                    documents = []
                    with self.track_time("batch"):
                        for line in batch.to_pylist():
                            document = self.get_document_from_dict(line, filepath, li)
                            if not document:
                                continue
                            documents.append(document)
                            li += 1
                    yield from documents
