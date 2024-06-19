"""
Used for n-gram decontamination.
First build an index using the tasks we want to use to decontaminate our training dataset.
Then read your training data and apply the filter with the index loaded.
"""

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, file_exists, get_datafolder, open_file
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.binaryio import read_np_from_file
from datatrove.utils.hashing import HashConfig, create_hash_func
from datatrove.utils.logging import logger
from datatrove.utils.text import TextNormConfig, ngrams, simplify_text
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


@dataclass
class NGramsDecontConfig:
    """
    Example for n_grams=4
    query = ['A', 'B', 'C', 'D', 'E'] (the prompt/instruction)
    label = ['F', 'G', 'H', 'I', 'J'] (the answer/gold)
    Will find the following N-GRAMS in the training data:
        'F G H I'
        'G H I J'
        + IF find_query_ngrams:
            'A B C D'
            'B C D E'
        + IF find_overlap_ngrams:
            'C D E F'
            'D E F G'
            'E F G H'
    """

    n_grams: int = 12
    find_query_ngrams: bool = False  # enable to also check for matches in n-grams containing only the input/prompt
    find_overlap_ngrams: bool = True  # will also find matches for n-grams containing BOTH input and query
    norm_config: TextNormConfig = field(default_factory=TextNormConfig)
    hash_config: HashConfig = field(default_factory=HashConfig)


class NGramsDecontIndexer(PipelineStep):
    """
    Creates a decontamination index (basically a list of uint64 hashes from ngrams) for each reference task.
    Ways to provide task data:
      - as input documents from the previous pipeline step with "text=label/correct answer"
        and metadata={"query": query/prompt/input, "task": task name}
      - as a list of strings in the format "suite|task" from the lighteval metadata table:
      https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks_table.jsonl as `lighteval_tasks`
      - a path to a text file containing one such list, with one "suite|task" per line as `lighteval_tasks`
      you can also define your custom tasks with `custom_lighteval_tasks`. See explanation for `custom_tasks` here:
      https://github.com/huggingface/lighteval/tree/main?tab=readme-ov-file#evaluate-a-model-on-extended-community-or-custom-tasks

    """

    type = "🦠 - DECONT"
    name = "💥 N-grams build index"
    _requires_dependencies = ["lighteval"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        lighteval_tasks: str | list[str] | None = None,  # list in the format suite|task or path to one such list
        custom_lighteval_tasks: str | None = None,
        config: NGramsDecontConfig = None,
        language: str = Languages.english,
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        # parse list of tasks
        if isinstance(lighteval_tasks, str):
            if file_exists(lighteval_tasks):
                with open_file(lighteval_tasks, "rt") as f:
                    self.lighteval_tasks = f.read().strip().splitlines()
            else:
                self.lighteval_tasks = [lighteval_tasks]
        else:
            self.lighteval_tasks = lighteval_tasks
        self.custom_lighteval_tasks = custom_lighteval_tasks
        self.config = config or NGramsDecontConfig()
        self.tokenizer = load_word_tokenizer(language)
        self.hash_func = create_hash_func(self.config.hash_config)

    def compute_hashes(self, label: str, query: str | None = None) -> list[int]:
        label_tokens = self.tokenizer.word_tokenize(simplify_text(label, self.config.norm_config))
        ngrams_to_compute = list(ngrams(label_tokens, self.config.n_grams))
        if query is not None:
            query_tokens = self.tokenizer.word_tokenize(simplify_text(query, self.config.norm_config))
            if self.config.find_query_ngrams:
                ngrams_to_compute.extend(ngrams(query_tokens, self.config.n_grams))
            if self.config.find_overlap_ngrams:
                # add tokens overlapping query and label
                """
                A, B, C, D, E | F, G, H, I, J
                5 grams
                B, C, D, E, F (-N + 1 + i:) + (:i + 1)
                ...
                E, F, G, H, I
                """
                ngrams_to_compute.extend(
                    [
                        query_tokens[-self.config.n_grams + 1 + i :] + label_tokens[: i + 1]
                        for i in range(self.config.n_grams - 1)
                        # make sure we actually get a list of size N
                        if len(query_tokens) >= self.config.n_grams - 1 - i and len(label_tokens) >= i + 1
                    ]
                )
        return list(map(self.hash_func, map(" ".join, ngrams_to_compute)))

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        if world_size != 1:
            raise ValueError("Decontamination index building requires a single worker.")
        hashes = defaultdict(set)
        # use whatever date is parsed in with the following format:
        # doc.text -> label
        # doc.metadata["input"] -> input
        if data:
            for doc in data:
                if not self.config.find_query_ngrams and "query" not in doc.metadata:
                    raise ValueError(
                        "only_label_ngrams is False but could not find 'query' field in documents metadata"
                    )
                hashes[doc.metadata.get("task", "input")].update(
                    self.compute_hashes(doc.text, doc.metadata.get("query", None))
                )

        # parse data from lighteval defined tasks
        from lighteval.tasks.lighteval_task import LightevalTask
        from lighteval.tasks.registry import Registry

        task_dict = Registry(cache_dir=os.getenv("HF_HOME")).get_task_dict(
            self.lighteval_tasks, custom_tasks=self.custom_lighteval_tasks
        )
        LightevalTask.load_datasets(task_dict.values())

        for task_name, task in task_dict.items():
            for eval_doc in task.eval_docs():
                try:
                    golds = eval_doc.get_golds()
                    query = eval_doc.query
                except Exception as e:
                    logger.warning(f"Error while fetching doc data: {e}")
                    continue
                for gold in golds:
                    hashes[task_name].update(self.compute_hashes(gold, query))

        for task_name, task_hashes in hashes.items():
            hashes_array = np.array(list(task_hashes), dtype=self.config.hash_config.np_descr)
            logger.info(f"Saving {len(task_hashes)} hashes for {task_name}")
            with self.output_folder.open(f"{task_name.replace(' ', '_')}.index.hashes", mode="wb") as f:
                if self.output_folder.is_local():
                    hashes_array.tofile(f)
                else:
                    f.write(hashes_array.tobytes())


class NGramsDecontFilter(BaseFilter):
    """
    Loads list of hashes created by the Indexer step.
    For each document in the block's input, we will check if any of its ngrams are part of the reference eval tasks.
    If so, they will be removed. The contaminated ngram and task where it was found will be saved in the removed
    document's metadata.
    """

    type = "🦠 - DECONT"
    name = "💥 N-grams decontaminate"

    def __init__(
        self,
        index_folder: DataFolderLike,
        config: NGramsDecontConfig = None,
        exclusion_writer: DiskWriter = None,
        language: str = Languages.english,
    ):
        super().__init__()
        self.index_folder = get_datafolder(index_folder)
        self.config = config or NGramsDecontConfig()
        self.exclusion_writer = exclusion_writer
        self.language = language
        self._index_hashes = None
        self.hash_func = create_hash_func(self.config.hash_config)
        self.tokenizer = load_word_tokenizer(language)

    def load_index_hashes(self):
        def load_index_from_file(file):
            with self.index_folder.open(file, mode="rb") as f:
                return file, read_np_from_file(
                    f, np.dtype(self.config.hash_config.np_descr), self.index_folder.is_local()
                ).tolist()

        with ThreadPoolExecutor() as pool:
            hashes = pool.map(load_index_from_file, self.index_folder.list_files())

        self._index_hashes = {}
        for filename, hashlist in hashes:
            taskname = filename.removesuffix(".index.hashes")
            logger.info(f"Loading {len(hashlist)} hashes for {taskname}")
            for hash in hashlist:
                self._index_hashes[hash] = taskname

    def filter(self, doc: Document) -> bool | Tuple[bool, str]:
        if self._index_hashes is None:
            self.load_index_hashes()

        text_tokens = self.tokenizer.word_tokenize(simplify_text(doc.text, self.config.norm_config))
        ngrams_to_compute = list(ngrams(text_tokens, self.config.n_grams))
        for n_gram in map(" ".join, ngrams_to_compute):
            task = self._index_hashes.get(self.hash_func(n_gram), None)
            if task is not None:
                doc.metadata["contaminated_ngram"] = n_gram
                doc.metadata["contaminated_task"] = task
                self.stat_update(f"contaminated_{task}")
                if ":" in task:
                    self.stat_update(f"contaminated_tg_{task[:task.index(':')]}")
                return False, "contaminated"
        return True
