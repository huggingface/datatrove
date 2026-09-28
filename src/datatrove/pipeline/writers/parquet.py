from collections import Counter, defaultdict
from typing import IO, Any, Callable, Literal

from datatrove.io import DataFolderLike
from datatrove.pipeline.writers.disk_base import DiskWriter


DEFAULT_CDC_OPTIONS = {"min_chunk_size": 256 * 1024, "max_chunk_size": 1024 * 1024, "norm_level": 0}


class ParquetWriter(DiskWriter):
    """Write data to datafolder (local or remote) in Parquet format

    Args:
        output_folder: a str, tuple or DataFolder where data should be saved
        output_filename: the filename to use when saving data, including extension. Can contain placeholders such as `${rank}` or metadata tags `${tag}`
        compression: parquet compression codec: "snappy" (default), "gzip", "brotli", "lz4", "zstd" or None
        adapter: a custom function to "adapt" the Document format to the desired output format
        batch_size: number of documents to buffer before writing a parquet batch
        expand_metadata: save each metadata entry in a different column instead of as a dictionary
        max_file_size: max size per parquet file, in bytes. Files rotate when they reach this size; a closed
            file is always readable, so smaller values lose less work if a job dies mid-run
        schema: an explicit `pyarrow.Schema` for the output. If None, the schema is inferred from the first
            document. Fields or types introduced later may be omitted or cause a schema mismatch, depending
            on batching. Pass an explicit schema when fields can be missing or heterogeneous
        save_media_bytes: save the media bytes column
        use_content_defined_chunking: write CDC-optimized parquet (better Xet deduplication on the Hub).
            True uses the default options; a dict is passed through to pyarrow
        write_page_index: write the parquet page index for faster filtered reads
    """

    default_output_filename: str = "${rank}.parquet"
    name = "📒 Parquet"
    _requires_dependencies = ["pyarrow"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = "snappy",
        adapter: Callable = None,
        batch_size: int = 1000,
        expand_metadata: bool = False,
        max_file_size: int = 5 * 2**30,  # 5GB
        schema: Any = None,
        save_media_bytes=False,
        use_content_defined_chunking=True,
        write_page_index=True,
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
            save_media_bytes=save_media_bytes,
        )
        self._writers = {}
        self._batches = defaultdict(list)
        self._file_counter = Counter()
        self.compression = compression
        self.batch_size = batch_size
        self.schema = schema
        # Write Optimized-parquet files
        # See https://huggingface.co/docs/hub/en/datasets-libraries#optimized-parquet-files
        if use_content_defined_chunking is True:
            use_content_defined_chunking = DEFAULT_CDC_OPTIONS
        self.use_content_defined_chunking = use_content_defined_chunking
        self.write_page_index = write_page_index

    def close_file(self, filename):
        """
            We need to write the last batch before closing the file
        Args:
            filename: filename to close
        """
        if self.max_file_size > 0 and filename not in self._writers:
            filename = self._get_filename_with_file_id(filename)
        self._write_batch(filename)
        if filename in self._writers:
            self._writers.pop(filename).close()
        super().close_file(filename)

    def _write_batch(self, filename):
        if not self._batches[filename]:
            return
        import pyarrow as pa

        # prepare batch
        batch = pa.RecordBatch.from_pylist(self._batches.pop(filename), schema=self.schema)
        # write batch
        self._writers[filename].write_batch(batch)

    def _write(self, document: dict, file_handler: IO, filename: str):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self.max_file_size > 0:
            filename = self._get_filename_with_file_id(filename)

        if filename not in self._writers:
            self._writers[filename] = pq.ParquetWriter(
                file_handler,
                schema=self.schema if self.schema is not None else pa.RecordBatch.from_pylist([document]).schema,
                compression=self.compression,
                use_content_defined_chunking=self.use_content_defined_chunking,
                write_page_index=self.write_page_index,
            )
        self._batches[filename].append(document)
        if len(self._batches[filename]) == self.batch_size:
            self._write_batch(filename)

    def close(self):
        for filename in list(self._writers.keys()):
            self.close_file(filename)
        self._batches.clear()
        self._writers.clear()
        super().close()
