import mmap

import numpy as np
from loguru import logger
from numpy.random import default_rng

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.merger import load_doc_ends


class DocumentTokenizerContextShuffler(PipelineStep):
    name = "🗃 Context Shuffler"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        window_size: int = 2048 + 1,
        seed: int = None,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.window_size = window_size
        self.rand = default_rng(seed)

    def get_ordering(self, all_doc_ends):
        doc_ids = np.concatenate([np.ones(len(doc_ends), dtype=int) * i for i, doc_ends in enumerate(all_doc_ends)])
        return self.rand.permutation(doc_ids)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        datafiles = self.input_folder.get_shard(rank, world_size, glob_pattern="*.ds")
        datafiles_index = self.input_folder.get_shard(rank, world_size, glob_pattern="*.ds.index")
        for datafile, index in zip(datafiles, datafiles_index):
            logger.info(f"Context shuffling {datafile.path} with a {self.window_size} token window")
            total_len = load_doc_ends(index)[-1]
            nr_windows = total_len // self.window_size
            ordering = self.rand.permutation(np.arange(0, nr_windows, dtype=int))
            with self.output_folder.open(datafile, "wb") as fout:
                with self.input_folder.open(datafile, "rb") as f:
                    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as unshuf:
                        with self.track_time():
                            for windowi in ordering:
                                start, end = windowi * self.window_size * 2, (windowi + 1) * self.window_size * 2
                                fout.write(unshuf[start:end])
