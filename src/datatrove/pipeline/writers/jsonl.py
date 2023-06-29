import dataclasses
import gzip
import json
from collections.abc import Callable

from datatrove.data import Document
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.io import OutputDataFile


class JsonlWriter(DiskWriter):
    open_fn: Callable = lambda self, f: gzip.open(f, mode='wt')
    default_output_filename: str = "${rank}.jsonl.gz"
    name = "🐿️ Jsonl"

    def _write(self, document: Document, file: OutputDataFile):
        file.file_handler.write(json.dumps(
            {
                key: val for key, val in
                dataclasses.asdict(document).items()
                if val
             },
            ensure_ascii=False) + "\n")
