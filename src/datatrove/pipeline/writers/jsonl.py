import dataclasses
import json
from typing import IO

from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.writers.disk_base import DiskWriter


class JsonlWriter(DiskWriter):
    default_output_filename: str = "${rank}.jsonl"
    name = "🐿 Jsonl"

    def __init__(self, output_folder: DataFolderLike, output_filename: str = None, compression: str | None = "gzip"):
        super().__init__(output_folder, output_filename=output_filename, compression=compression)

    def _write(self, document: Document, file: IO):
        file.write(
            json.dumps({key: val for key, val in dataclasses.asdict(document).items() if val}, ensure_ascii=False)
            + "\n"
        )
