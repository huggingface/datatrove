from abc import abstractmethod
from contextlib import nullcontext
from typing import Callable

from loguru import logger
from tqdm import tqdm

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import BaseInputDataFile, BaseInputDataFolder
from datatrove.pipeline.base import PipelineStep


class BaseReader(PipelineStep):
    type = "📖 - READER"

    def __init__(
        self,
        data_folder: BaseInputDataFolder,
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        content_key: str = "content",
        id_key: str = "data_id",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_folder = data_folder
        self.limit = limit
        self.progress = progress
        self.content_key = content_key
        self.id_key = id_key
        self.adapter = adapter if adapter else self._default_adapter
        self.empty_warning = False

    def _default_adapter(self, data: dict, path: str, id_in_file: int):
        return {
            "content": data.pop(self.content_key, ""),
            "data_id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": data,
        }

    def get_document_from_dict(self, data: dict, path: str, id_in_file: int):
        parsed_data = self.adapter(data, path, id_in_file)
        if not parsed_data.get("content", None):
            if not self.empty_warning:
                self.empty_warning = True
                logger.warning("Found document without content, skipping.")
            return None
        document = Document(**parsed_data)
        document.metadata.setdefault("file_path", path)
        return document

    @abstractmethod
    def read_file(self, datafile: BaseInputDataFile):
        raise NotImplementedError

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.data_folder.set_lock(dl_lock)

    def read_files_shard(self, shard):
        li = 0
        with tqdm(total=self.limit if self.limit != -1 else None) if self.progress else nullcontext() as pbar:
            for datafile in shard:
                logger.info(f"Reading input file {datafile.path}")
                for document in self.read_file(datafile):
                    if self.limit != -1 and li >= self.limit:
                        return
                    yield document
                    if self.progress:
                        pbar.update()
                    li += 1

    def __call__(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data
        yield from self.read_files_shard(self.data_folder.get_files_shard(rank, world_size))
