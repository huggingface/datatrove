import queue
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from threading import Event
from typing import BinaryIO

import numpy as np
from loguru import logger
from numpy.random import default_rng

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.tokenizer import TokenizedFile


class DocumentTokenizerMerger(PipelineStep):
    """ """

    name = "🗃 Merger"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        save_filename: str,  # if defined, the final output filename will be this
        max_tokens_per_file: int = 100e9,  # max number of tokens per file. default: 100GT
        max_tokens: int = -1,  # max number of tokens to process
        shuffle: bool = True,  # whether to shuffle documents in the dataset
        upload_block_size: int = 20 * 2**20,  # 20MB
        # upload_block_size * 10000 must be bigger than 2*max_tokens_per_file,
        # or s3 uploads will fail
        seed: int = None,
        save_loss_metadata: bool = False,
        save_final_metadata: bool = True,
        read_queue_size: int = 5,  # how many documents to preload from each file
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.save_filename = save_filename
        self.max_tokens_per_file = max_tokens_per_file
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.save_loss_metadata = save_loss_metadata
        self.rand = default_rng(seed)
        self.save_final_metadata = save_final_metadata
        self.upload_block_size = upload_block_size
        self.read_queue_size = read_queue_size

    def get_ordering(self, all_doc_ends):
        """

        Args:
          all_doc_ends:

        Returns:

        """
        doc_ids = np.concatenate([np.ones(len(doc_ends), dtype=int) * i for i, doc_ends in enumerate(all_doc_ends)])
        return doc_ids if not self.shuffle else self.rand.permutation(doc_ids)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """

        Args:
          data: DocumentsPipeline:  (Default value = None)
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:

        """
        assert world_size == 1, "world_size must be 1 for DocumentTokenizerMerger"
        datafiles = self.input_folder.list_files(glob_pattern="*.ds")
        datafiles_index = self.input_folder.list_files(glob_pattern="*.ds.index")
        datafiles_loss = (
            self.input_folder.list_files(glob_pattern="*.ds.loss")
            if self.save_loss_metadata
            else ([None] * len(datafiles))
        )
        assert len(datafiles) == len(datafiles_index) == len(datafiles_loss), (
            f"Mismatch between number of .ds, "
            ".ds.index and/or .ds.loss files"
            f"({len(datafiles)} vs {len(datafiles_index)} vs {len(datafiles_loss)})"
        )

        if len(datafiles) == 0:
            logger.error("No datafiles found!")
            sys.exit(-1)

        doc_ends = [load_doc_ends(self.input_folder.open(file, "rb")) for file in datafiles_index]
        token_inputs = list(
            map(partial(get_data_reader, nb_bytes=2), self.input_folder.open_files(datafiles), doc_ends)
        )
        loss_inputs = (
            list(map(partial(get_data_reader, nb_bytes=1), self.input_folder.open_files(datafiles_loss), doc_ends))
            if self.save_loss_metadata
            else None
        )

        tokenizer_name = None
        if self.save_final_metadata:
            if self.input_folder.isfile(f"{datafiles[0]}.metadata"):
                with self.input_folder.open(f"{datafiles[0]}.metadata", "rt") as f:
                    tokenizer_name = f.read().splitlines()[0]

        ordering = self.get_ordering(doc_ends)

        def input_data_read_worker(token_input, loss_input, results_queue, size, early_stop: Event):
            for _ in range(size):
                if early_stop.is_set():
                    break
                # with the loop we should actually error out of there is an issue related to the size
                read_tokens = next(token_input)
                loss = None
                if loss_input:
                    loss = next(loss_input)
                results_queue.put((read_tokens, loss))

        with ThreadPoolExecutor() as executor:
            input_queues = [queue.Queue(self.read_queue_size) for _ in range(len(token_inputs))]
            early_stop = Event()
            workers = [
                executor.submit(input_data_read_worker, token_input, loss_input, q, len(docs), early_stop)
                for token_input, loss_input, q, docs in zip(token_inputs, loss_inputs, input_queues, doc_ends)
            ]

            file_ct = 0
            output_file = TokenizedFile(
                output_folder=self.output_folder,
                filename=f"{file_ct:03d}_{self.save_filename}.ds",
                save_loss_metadata=self.save_loss_metadata,
                upload_block_size=self.upload_block_size,
            )
            for input_file_id in ordering:
                if 0 < self.max_tokens <= self.stats["tokens"].total:
                    early_stop.set()
                    for inp_q in input_queues:
                        # safe because there are no other queue consumers
                        if inp_q.full():
                            # free up a spot so that the event can be checked
                            inp_q.get_nowait()
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
                        upload_block_size=self.upload_block_size,
                    )
                # copy tokens and loss
                tokens_data, loss_data = input_queues[input_file_id].get()
                output_file.write_bytes(tokens_data)
                if loss_data:
                    output_file.write_loss_bytes(loss_data)
                self.stat_update("tokens", value=len(tokens_data) // 2)
            # cleanup
            output_file.close()
            if self.save_final_metadata:
                output_file.save_final_metadata(tokenizer_name)
                # save final total metadata file
                output_file.save_final_metadata(
                    tokenizer_name, self.stats["tokens"].total, filename=f"{self.save_filename}.ds"
                )
            output_file.close()
            wait(workers)


def load_doc_ends(file: BinaryIO):
    with file as f:
        return np.frombuffer(f.read(), dtype=np.uint64).tolist()


def get_data_reader(file: BinaryIO, doc_ends: list, nb_bytes: int):
    with file as f:
        start_e = 0
        for r_e in doc_ends:
            yield f.read((r_e - start_e) * nb_bytes)
            start_e = r_e
