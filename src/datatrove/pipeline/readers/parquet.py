from typing import Callable

from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


class ParquetReader(BaseDiskReader):
    """Read data from Parquet files.
        Will read each batch as a separate document.

    Args:
        data_folder: a str, tuple or DataFolder object representing a path/filesystem
        paths_file: optionally provide a file with one path per line (without the `data_folder` prefix) to read.
        limit: limit the number of documents to read. Useful for debugging
        skip: skip the first n rows
        batch_size: the batch size to use (default: 1000)
        read_metadata: if True, will read the metadata (default: True)
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

    name = "📒 Parquet"
    _requires_dependencies = ["pyarrow"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
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
