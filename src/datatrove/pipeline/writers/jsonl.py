import dataclasses
import gzip
import json
from collections.abc import Callable

from datatrove.data import Document
from datatrove.io import OutputDataFile
from datatrove.pipeline.writers.disk_base import DiskWriter


class JsonlWriter(DiskWriter):
    def _gzip_open(self, f):
        return gzip.open(f, mode="wt")

    open_fn: Callable = _gzip_open
    default_output_filename: str = "${rank}.jsonl.gz"
    name = "🐿️ Jsonl"

    def _write(self, document: Document, file: OutputDataFile):
        file.file_handler.write(
            json.dumps({key: val for key, val in dataclasses.asdict(document).items() if val}, ensure_ascii=False)
            + "\n"
        )
