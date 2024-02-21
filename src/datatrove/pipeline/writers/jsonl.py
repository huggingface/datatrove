import json
from typing import IO, Callable

from datatrove.io import DataFolderLike
from datatrove.pipeline.writers.disk_base import DiskWriter


class JsonlWriter(DiskWriter):
    default_output_filename: str = "${rank}.jsonl"
    name = "🐿 Jsonl"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: str | None = "gzip",
        adapter: Callable = None,
    ):
        super().__init__(output_folder, output_filename=output_filename, compression=compression, adapter=adapter)

    def _write(self, document: dict, file_handler: IO, _filename: str):
        file_handler.write(json.dumps(document, ensure_ascii=False) + "\n")
