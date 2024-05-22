import csv
from typing import Callable, Literal

from datatrove.io import DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


class CsvReader(BaseDiskReader):
    """Read data from CSV files.
        Will read each line as a separate document.

    Args:
        data_folder: the data folder to read from
        compression: the compression to use (default: "infer")
        limit: limit the number of CSV lines to read in each rank. Useful for debugging
        skip: skip the first n rows
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

    name = "🔢 Csv"

    def __init__(
        self,
        data_folder: DataFolderLike,
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        skip: int = 0,
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
        self.compression = compression
        self.empty_warning = False

    def read_file(self, filepath: str):
        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            csv_reader = csv.DictReader(f)
            for di, d in enumerate(csv_reader):
                with self.track_time():
                    document = self.get_document_from_dict(d, filepath, di)
                    if not document:
                        continue
                yield document


CSVReader = CsvReader
