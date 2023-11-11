import json
from json import JSONDecodeError
from typing import Literal

from loguru import logger

from datatrove.io import BaseInputDataFile, BaseInputDataFolder
from datatrove.pipeline.readers.base import BaseReader


class JsonlReader(BaseReader):
    name = "🐿 Jsonl"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        compression: Literal["guess", "gzip", "zst"] | None = "guess",
        **kwargs,
    ):
        super().__init__(data_folder, **kwargs)
        self.compression = compression

    def read_file(self, datafile: BaseInputDataFile):
        with datafile.open(compression=self.compression) as f:
            for li, line in enumerate(f):
                with self.track_time():
                    try:
                        document = self.get_document_from_dict(json.loads(line), datafile, li)
                        if not document:
                            continue
                    except (EOFError, JSONDecodeError) as e:
                        logger.warning(f"Error when reading `{datafile.path}`: {e}")
                        continue
                yield document
