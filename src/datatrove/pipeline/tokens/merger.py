import mmap

import numpy as np
from numpy.random import default_rng

from datatrove.data import DocumentsPipeline
from datatrove.io import BaseInputDataFolder, BaseOutputDataFolder, InputDataFile
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.tokenizer import TokenizedFile


class DocumentTokenizerMerger(PipelineStep):
    def __init__(
        self,
        input_folder: BaseInputDataFolder,
        output_folder: BaseOutputDataFolder,
        save_filename: str,  # if defined, the final output filename will be this
        max_tokens: int = 100e9,  # max number of tokens per file. default: 100GT
        shuffle: bool = True,  # whether to shuffle documents in the dataset
        save_loss_metadata: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.save_filename = save_filename
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.save_loss_metadata = save_loss_metadata
        self.rand = default_rng()

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.input_folder.set_lock(dl_lock)
        self.output_folder.set_lock(up_lock)

    def get_ordering(self, all_doc_ends):
        doc_ids = np.concatenate([np.ones(len(doc_ends), dtype=int) * i for i, doc_ends in enumerate(all_doc_ends)])
        return doc_ids if not self.shuffle else self.rand.permutation(doc_ids)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        assert world_size == 1, "world_size must be 1 for DocumentTokenizerMerger"
        datafiles = self.input_folder.list_files(extension=".ds")
        datafiles_index = self.input_folder.list_files(extension=".ds.index")
        datafiles_loss = (
            self.input_folder.list_files(extension=".ds.loss") if self.save_loss_metadata else [None] * len(datafiles)
        )
        assert len(datafiles) == len(datafiles_index) == len(datafiles_loss), (
            "Mismatch between number of .ds, " ".ds.index and/or .ds.loss files"
        )

        doc_ends = [load_doc_ends(file) for file in datafiles_index]
        token_inputs = list(map(load_input_mmap, datafiles))
        loss_inputs = list(map(load_input_mmap, datafiles_loss)) if self.save_loss_metadata else datafiles_loss
        ordering = self.get_ordering(doc_ends)
        read_idx = np.zeros(len(datafiles), dtype=int)

        file_ct = 0
        output_file = TokenizedFile(
            output_folder=self.output_folder,
            filename=f"{file_ct:03d}_{self.save_filename}.ds",
            save_loss_metadata=self.save_loss_metadata,
        )
        output_file.open()
        for input_file_id in ordering:
            if 0 < self.max_tokens <= len(output_file):
                output_file.close()
                file_ct += 1
                output_file = TokenizedFile(
                    output_folder=self.output_folder,
                    filename=f"{file_ct:03d}_{self.save_filename}.ds",
                    save_loss_metadata=self.save_loss_metadata,
                )
                output_file.open()
            start, end = (
                doc_ends[input_file_id][read_idx[input_file_id] - 1] if read_idx[input_file_id] > 0 else 0,
                doc_ends[input_file_id][read_idx[input_file_id]],
            )
            # copy tokens and loss
            output_file.write_bytes(token_inputs[input_file_id][start * 2 : end * 2])
            output_file.write_loss_bytes(loss_inputs[input_file_id][start:end])
            read_idx[input_file_id] += 1
        # cleanup
        for token_input, loss_input in zip(token_inputs, loss_inputs):
            token_input.close()
            if loss_input:
                loss_input.close()
        output_file.close()
        self.output_folder.close()


def load_doc_ends(file: InputDataFile):
    with file.open_binary() as f:
        return np.frombuffer(f.read(), dtype=np.uint32)


def load_input_mmap(file: InputDataFile):
    with file.open_binary() as f:
        return mmap.mmap(f.fileno(), 0)
