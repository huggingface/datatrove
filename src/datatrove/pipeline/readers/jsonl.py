import json
from json import JSONDecodeError
from typing import Callable, Literal

from loguru import logger

from datatrove.data import Document
from datatrove.io import BaseInputDataFolder, InputDataFile
from datatrove.pipeline.readers.base import BaseReader


class JsonlReader(BaseReader):
    name = "🐿 Jsonl"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        compression: Literal["gzip", "zst"] | None = None,
        adapter: Callable = None,
        content_key: str = "content",
        **kwargs,
    ):
        super().__init__(data_folder, **kwargs)
        self.compression = compression
        self.content_key = content_key
        self.adapter = adapter if adapter else lambda d, path, li: d
        self.empty_warning = False

    def read_file(self, datafile: InputDataFile):
        with datafile.open(compression=self.compression) as f:
            for li, line in enumerate(f):
                with self.stats.time_manager:
                    try:
                        d = json.loads(line)
                        if not d.get(self.content_key, None):
                            if not self.empty_warning:
                                self.empty_warning = True
                                logger.warning("Found document without content, skipping.")
                            continue
                        document = Document(**self.adapter(d, datafile.path, li))
                        document.metadata.setdefault("file_path", datafile.path)
                    except (EOFError, JSONDecodeError) as e:
                        logger.warning(f"Error when reading `{datafile.path}`: {e}")
                        continue
                yield document
