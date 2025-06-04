from typing import IO, Callable
from datatrove.data import Document, DocumentsPipeline, Media
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
import uuid
from datatrove.utils.typeshelper import StatHints
import os


class TempDirWriter(PipelineStep):
    def __init__(self, temp_df: DataFolderLike):
        self.temp_df = get_datafolder(temp_df)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            media_paths = []
            for media in doc.media:
                if media.media_bytes is not None:
                    unique_id = uuid.uuid4()  # Generate a unique identifier
                    with self.temp_df.open(f"{rank}/{unique_id}.pdf", "wb") as f:
                        f.write(media.media_bytes)
                    media_paths.append(f.path)
                    media.metadata["local_path"] = f.path
                    media.media_bytes = None
            yield doc
            # delete the files
            for path in media_paths:
                os.remove(path)