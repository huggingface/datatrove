import dataclasses
import json
from string import Template

from datatrove.data import Document
from datatrove.io import BaseOutputDataFile, BaseOutputDataFolder
from datatrove.pipeline.writers.disk_base import DiskWriter


class JsonlWriter(DiskWriter):
    default_output_filename: str = "${rank}.jsonl"
    name = "🐿 Jsonl"

    def __init__(self, output_folder: BaseOutputDataFolder, output_filename: str = None, gzip: bool = True):
        super().__init__(output_folder, output_filename=output_filename)
        self.gzip = gzip
        if self.gzip:
            self.output_filename = Template(self.output_filename.template + ".gz")

    def open(self, output_filename):
        return self.output_folder.open(output_filename, mode="wt", gzip=self.gzip)

    def _write(self, document: Document, file: BaseOutputDataFile):
        file.write(
            json.dumps({key: val for key, val in dataclasses.asdict(document).items() if val}, ensure_ascii=False)
            + "\n"
        )
