import mmap

import numpy as np
from numpy.random import default_rng

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.merger import load_doc_ends
from datatrove.utils.logging import logger


class DocumentTokenizerContextShuffler(PipelineStep):
    """Shuffles a .ds file on the context length level. This block will move around windows of `window_size` tokens.

    Args:
        input_folder: the input folder to read the tokenized documents from
        output_folder: the output folder to write the shuffled documents to
        window_size: the size of the window to shuffle (default: 2048 + 1)
        seed: the seed for the random number generator (default: None)
        token_size (int): size of each token, in bytes
    """

    name = "🗃 Context Shuffler"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        window_size: int = 2048 + 1,
        seed: int = None,
        token_size: int = 2,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.window_size = window_size
        self.token_size = token_size
        self.rand = default_rng(seed)

    def get_ordering(self, all_doc_ends):
        """
            Computes the new ordering of context windows

        Args:
          all_doc_ends:

        Returns:

        """
        doc_ids = np.concatenate([np.ones(len(doc_ends), dtype=int) * i for i, doc_ends in enumerate(all_doc_ends)])
        return self.rand.permutation(doc_ids)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """

        Args:
          data: DocumentsPipeline:  (Default value = None)
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:

        """
        datafiles = self.input_folder.get_shard(rank, world_size, glob_pattern="*.ds")
        datafiles_index = self.input_folder.get_shard(rank, world_size, glob_pattern="*.ds.index")
        for datafile, index in zip(datafiles, datafiles_index):
            logger.info(f"Context shuffling {datafile.path} with a {self.window_size} token window")
            total_len = load_doc_ends(index)[-1]
            nr_windows = total_len // self.window_size
            ordering = self.rand.permutation(np.arange(0, nr_windows, dtype=int))
            with self.output_folder.open(datafile, "wb") as fout:
                with self.input_folder.open(datafile, "rb") as f:
                    # TODO: replace mmap implementation which only works locally
                    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as unshuf:
                        with self.track_time():
                            for windowi in ordering:
                                start, end = (
                                    windowi * self.window_size * self.token_size,
                                    (windowi + 1) * self.window_size * self.token_size,
                                )
                                fout.write(unshuf[start:end])
