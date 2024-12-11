import csv
import os
import urllib.request

import numpy as np
from huggingface_hub import cached_assets_path

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.logging import logger
from datatrove.utils.text import split_into_words
from datatrove.utils.typeshelper import Languages


UNIGRAM_DOWNLOAD = "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/lucas/google-1T-unigram/unigram_freq.csv"


class UnigramLogProbFilter(BaseFilter):
    """
    Computes average unigram log probability based on word frequencies from
    https://www.kaggle.com/datasets/rtatman/english-word-frequency

    Idea taken from https://huggingface.co/datasets/allenai/peS2o
    """

    name = "🧑‍🍳 Unigram log-prob filter"

    def __init__(
        self, logprobs_threshold: float = -10, exclusion_writer: DiskWriter = None, language: str = Languages.english
    ):
        """

        Args:
            logprobs_threshold: the minimum average unigram logprobs needed to keep a document
            exclusion_writer:
        """
        super().__init__(exclusion_writer)
        self.logprobs_threshold = logprobs_threshold
        self.unigram_frequencies = self.get_frequencies()
        self.language = language

    def get_frequencies(self):
        download_dir = cached_assets_path(
            library_name="datatrove", namespace="filters", subfolder="unigram_logprob_filter"
        )
        unigram_freq_file = os.path.join(download_dir, "unigram_freq.csv")
        if not os.path.isfile(unigram_freq_file):
            logger.info("⬇️ Downloading unigram-frequencies ...")
            urllib.request.urlretrieve(UNIGRAM_DOWNLOAD, unigram_freq_file)

        words = []
        counts = []
        with open(unigram_freq_file, encoding="utf-8", newline="") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                words.append(row["word"])
                counts.append(int(row["count"]))
        total_count = sum(counts)
        return {word: count / total_count for word, count in zip(words, counts)}

    def get_logprob(self, doc):
        words = split_into_words(doc.text, self.language)
        freqs = [self.unigram_frequencies.get(word.lower(), 1e-9) for word in words]

        if len(freqs) == 0:
            return 0
        return sum([np.log(f) for f in freqs]) / len(freqs)

    def filter(self, doc: Document) -> bool:
        """
            Checks if the average unigram probability is above the threshold. This assumes the text is in english.
        Args:
            doc:

        Returns:

        """
        return self.get_logprob(doc) > self.logprobs_threshold
