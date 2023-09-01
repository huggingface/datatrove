import mmap

import numpy as np
from loguru import logger
from numpy.random import default_rng

from datatrove.data import DocumentsPipeline
from datatrove.io import BaseInputDataFolder, BaseOutputDataFolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.merger import load_doc_ends


class DocumentTokenizerContextShuffler(PipelineStep):
    name = "🗃 Context Shuffler"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        input_folder: BaseInputDataFolder,
        output_folder: BaseOutputDataFolder,
        window_size: int = 2048 + 1,
        seed: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.window_size = window_size
        self.rand = default_rng(seed)

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.input_folder.set_lock(dl_lock)
        self.output_folder.set_lock(up_lock)

    def get_ordering(self, all_doc_ends):
        doc_ids = np.concatenate([np.ones(len(doc_ends), dtype=int) * i for i, doc_ends in enumerate(all_doc_ends)])
        return doc_ids if not self.shuffle else self.rand.permutation(doc_ids)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        datafiles = self.input_folder.get_files_shard(rank, world_size, extension=".ds")
        datafiles_index = self.input_folder.get_files_shard(rank, world_size, extension=".ds.index")
        for datafile, index in zip(datafiles, datafiles_index):
            logger.info(f"Context shuffling {datafile.path} with a {self.window_size} token window")
            total_len = load_doc_ends(index)[-1]
            nr_windows = total_len // self.window_size
            ordering = self.rand.permutation(np.arange(0, nr_windows, dtype=int))
            output_file = self.output_folder.create_new_file(datafile.relative_path)
            with datafile.open_binary() as f:
                with mmap.mmap(f.fileno(), 0) as unshuf:
                    with output_file.open("wb") as fout:
                        for windowi in ordering:
                            start, end = windowi * self.window_size * 2, (windowi + 1) * self.window_size * 2
                            fout.write(unshuf[start:end])
        self.output_folder.close()
