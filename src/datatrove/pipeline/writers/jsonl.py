import dataclasses
import json

from datatrove.data import Document
from datatrove.io import OutputDataFile
from datatrove.pipeline.writers.disk_base import DiskWriter


class JsonlWriter(DiskWriter):
    default_output_filename: str = "${rank}.jsonl.gz"
    name = "🐿️ Jsonl"

    def open(self, output_filename):
        return self.output_folder.open(output_filename, mode="wt", gzip=True)

    def _write(self, document: Document, file: OutputDataFile):
        file.file_handler.write(
            json.dumps({key: val for key, val in dataclasses.asdict(document).items() if val}, ensure_ascii=False)
            + "\n"
        )
