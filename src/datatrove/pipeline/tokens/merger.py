from functools import partial

import numpy as np
from numpy.random import default_rng

from datatrove.data import DocumentsPipeline
from datatrove.io import BaseInputDataFile, BaseInputDataFolder, BaseOutputDataFolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.tokenizer import TokenizedFile


class DocumentTokenizerMerger(PipelineStep):
    name = "🗃 Merger"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        input_folder: BaseInputDataFolder,
        output_folder: BaseOutputDataFolder,
        save_filename: str,  # if defined, the final output filename will be this
        max_tokens_per_file: int = 100e9,  # max number of tokens per file. default: 100GT
        max_tokens: int = -1,  # max number of tokens to process
        shuffle: bool = True,  # whether to shuffle documents in the dataset
        seed: int = None,
        save_loss_metadata: bool = False,
        save_final_metadata: bool = True,
    ):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.save_filename = save_filename
        self.max_tokens_per_file = max_tokens_per_file
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.save_loss_metadata = save_loss_metadata
        self.rand = default_rng(seed)
        self.save_final_metadata = save_final_metadata

    def set_up_dl_locks(self, dl_lock, up_lock):
        self.input_folder.set_lock(dl_lock)
        self.output_folder.set_lock(up_lock)

    def get_ordering(self, all_doc_ends):
        doc_ids = np.concatenate([np.ones(len(doc_ends), dtype=int) * i for i, doc_ends in enumerate(all_doc_ends)])
        return doc_ids if not self.shuffle else self.rand.permutation(doc_ids)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        assert world_size == 1, "world_size must be 1 for DocumentTokenizerMerger"
        datafiles = self.input_folder.list_files(extension=".ds")
        datafiles_index = self.input_folder.list_files(extension=".ds.index")
        datafiles_loss = (
            self.input_folder.list_files(extension=".ds.loss") if self.save_loss_metadata else [None] * len(datafiles)
        )
        assert len(datafiles) == len(datafiles_index) == len(datafiles_loss), (
            f"Mismatch between number of .ds, "
            ".ds.index and/or .ds.loss files"
            f"({len(datafiles)} vs {len(datafiles_index)} vs {len(datafiles_loss)})"
        )

        doc_ends = [load_doc_ends(file) for file in datafiles_index]
        token_inputs = list(map(partial(get_data_reader, nb_bytes=2), datafiles, doc_ends))
        loss_inputs = (
            list(map(partial(get_data_reader, nb_bytes=1), datafiles_loss, doc_ends))
            if self.save_loss_metadata
            else None
        )

        tokenizer_name = None
        if self.save_final_metadata:
            first_metadata_file = self.input_folder.get_file(f"{datafiles[0].relative_path}.metadata")
            if first_metadata_file:
                with first_metadata_file.open() as f:
                    tokenizer_name = f.read().splitlines()[0]

        ordering = self.get_ordering(doc_ends)

        file_ct = 0
        output_file = TokenizedFile(
            output_folder=self.output_folder,
            filename=f"{file_ct:03d}_{self.save_filename}.ds",
            save_loss_metadata=self.save_loss_metadata,
        )
        for input_file_id in ordering:
            if 0 < self.max_tokens <= self.stats["tokens"].total:
                break
            if 0 < self.max_tokens_per_file <= len(output_file):
                output_file.close()
                file_ct += 1
                if self.save_final_metadata:
                    output_file.save_final_metadata(tokenizer_name)
                output_file = TokenizedFile(
                    output_folder=self.output_folder,
                    filename=f"{file_ct:03d}_{self.save_filename}.ds",
                    save_loss_metadata=self.save_loss_metadata,
                )
            # copy tokens and loss
            tokens = next(token_inputs[input_file_id])
            output_file.write_bytes(tokens)
            if loss_inputs:
                output_file.write_loss_bytes(next(loss_inputs[input_file_id]))
            self.stat_update("tokens", value=len(tokens) // 2)
        # cleanup
        output_file.close()
        if self.save_final_metadata:
            output_file.save_final_metadata(tokenizer_name)
            # save final total metadata file
            output_file.save_final_metadata(
                tokenizer_name, self.stats["tokens"].total, filename=f"{self.save_filename}.ds"
            )
        self.output_folder.close()


def load_doc_ends(file: BaseInputDataFile):
    with file.open_binary() as f:
        return np.frombuffer(f.read(), dtype=np.uint64).tolist()


def get_data_reader(file: BaseInputDataFile, doc_ends: list, nb_bytes: int):
    with file.open_binary() as f:
        start_e = 0
        for r_e in doc_ends:
            yield f.read((r_e - start_e) * nb_bytes)
            start_e = r_e
