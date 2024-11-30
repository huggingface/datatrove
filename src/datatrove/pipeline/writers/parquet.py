from collections import Counter, defaultdict
from typing import IO, Callable, Literal

from datatrove.io import DataFolderLike
from datatrove.pipeline.writers.disk_base import DiskWriter


def parquet_fix(document: dict) -> dict:
    """
    You can create your own adapter that returns a dictionary in your preferred format,
    while addressing the Parquet issue with empty struct fields.

    Args:
        document: document to format

    Returns: a dictionary to write to disk
    """

    def ensure_non_empty_structs(data):
        """
        Recursively ensure that any dictionary which would correspond to an empty struct
        in Parquet has a dummy field added.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively fix nested dictionaries
                data[key] = ensure_non_empty_structs(value)
                # Add a dummy field if the dictionary is empty
                if not data[key]:
                    data[key] = {"__dummy_field": None}
        return data

    data = {key: val for key, val in document.items() if val}

    # if self.expand_metadata and "metadata" in data:
    #     data |= data.pop("metadata")

    # Fix empty structs in the dictionary
    return ensure_non_empty_structs(data)


class ParquetWriter(DiskWriter):
    default_output_filename: str = "${rank}.parquet"
    name = "📒 Parquet"
    _requires_dependencies = ["pyarrow"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = None,
        adapter: Callable = None,
        batch_size: int = 1000,
        expand_metadata: bool = False,
        max_file_size: int = 5 * 2**30,  # 5GB
    ):
        # Validate the compression setting
        if compression not in {"snappy", "gzip", "brotli", "lz4", "zstd", None}:
            raise ValueError(
                "Invalid compression type. Allowed types are 'snappy', 'gzip', 'brotli', 'lz4', 'zstd', or None."
            )

        super().__init__(
            output_folder,
            output_filename,
            compression=None,  # Ensure superclass initializes without compression
            adapter=adapter,
            mode="wb",
            expand_metadata=expand_metadata,
            max_file_size=max_file_size,
        )
        self._writers = {}
        self._batches = defaultdict(list)
        self._file_counter = Counter()
        self.compression = compression
        self.batch_size = batch_size

    def _on_file_switch(self, original_name, old_filename, new_filename):
        """
            Called when we are switching file from "old_filename" to "new_filename" (original_name is the filename
            without 000_, 001_, etc)
        Args:
            original_name: name without file counter
            old_filename: old full filename
            new_filename: new full filename
        """
        self._writers.pop(original_name).close()
        super()._on_file_switch(original_name, old_filename, new_filename)

    def _write_batch(self, filename):
        if not self._batches[filename]:
            return
        import pyarrow as pa

        # prepare batch
        _batch = self._batches.pop(filename)
        batch = pa.RecordBatch.from_pylist(_batch)
        # write batch
        self._writers[filename].write_batch(batch)

    def _write(self, document: dict, file_handler: IO, filename: str):
        import pyarrow as pa
        import pyarrow.parquet as pq

        document = parquet_fix(document)

        if filename not in self._writers:
            self._writers[filename] = pq.ParquetWriter(
                file_handler,
                schema=pa.RecordBatch.from_pylist([document]).schema,
                compression=self.compression,
            )
        self._batches[filename].append(document)
        if len(self._batches[filename]) == self.batch_size:
            self._write_batch(filename)

    def close(self):
        for filename in list(self._batches.keys()):
            self._write_batch(filename)
        for writer in self._writers.values():
            writer.close()
        self._batches.clear()
        self._writers.clear()
        super().close()
