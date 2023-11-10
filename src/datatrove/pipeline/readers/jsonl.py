import json
from json import JSONDecodeError
from typing import Callable, Literal

from loguru import logger

from datatrove.data import Document
from datatrove.io import BaseInputDataFile, BaseInputDataFolder
from datatrove.pipeline.readers.base import BaseReader


class JsonlReader(BaseReader):
    name = "🐿 Jsonl"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        compression: Literal["guess", "gzip", "zst"] | None = "guess",
        adapter: Callable = None,
        content_key: str = "content",
        id_key: str = "data_id",
        **kwargs,
    ):
        super().__init__(data_folder, **kwargs)
        self.compression = compression
        self.content_key = content_key
        self.id_key = id_key
        self.adapter = adapter if adapter else self._default_adapter
        self.empty_warning = False

    def _default_adapter(self, d: dict, path: str, li: int):
        return {
            "content": d.pop(self.content_key, ""),
            "data_id": d.pop(self.id_key, f"{path}/{li}"),
            "media": d.pop("media", []),
            "metadata": d,
        }

    def read_file(self, datafile: BaseInputDataFile):
        with datafile.open(compression=self.compression) as f:
            for li, line in enumerate(f):
                with self.stats.time_manager:
                    try:
                        d = self.adapter(json.loads(line), datafile.path, li)
                        if not d.get("content", None):
                            if not self.empty_warning:
                                self.empty_warning = True
                                logger.warning("Found document without content, skipping.")
                            continue
                        document = Document(**d)
                        document.metadata.setdefault("file_path", datafile.path)
                    except (EOFError, JSONDecodeError) as e:
                        logger.warning(f"Error when reading `{datafile.path}`: {e}")
                        continue
                yield document
