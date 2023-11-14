import json
from json import JSONDecodeError
from typing import Callable, Literal

from loguru import logger

from datatrove.io import BaseInputDataFile, BaseInputDataFolder
from datatrove.pipeline.readers.base import BaseReader


class JsonlReader(BaseReader):
    name = "🐿 Jsonl"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        compression: Literal["guess", "gzip", "zst"] | None = "guess",
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        content_key: str = "content",
        id_key: str = "data_id",
    ):
        super().__init__(data_folder, limit, progress, adapter, content_key, id_key)
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
