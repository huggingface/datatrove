import json
from json import JSONDecodeError
from typing import Callable, Literal

from loguru import logger

from datatrove.io import DataFolderLike
from datatrove.pipeline.readers.base import BaseReader


class JsonlReader(BaseReader):
    name = "🐿 Jsonl"

    def __init__(
        self,
        data_folder: DataFolderLike,
        compression: Literal["guess", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
    ):
        super().__init__(data_folder, limit, progress, adapter, text_key, id_key, default_metadata)
        self.compression = compression

    def read_file(self, filepath: str):
        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            for li, line in enumerate(f):
                with self.track_time():
                    try:
                        document = self.get_document_from_dict(json.loads(line), filepath, li)
                        if not document:
                            continue
                    except (EOFError, JSONDecodeError) as e:
                        logger.warning(f"Error when reading `{filepath}`: {e}")
                        continue
                yield document
