import csv
import os
import urllib.request

import numpy as np
from huggingface_hub import cached_assets_path
from loguru import logger

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


UNIGRAM_DOWNLOAD = "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/lucas/google-1T-unigram/unigram_freq.csv"


class UnigramLogProbFilter(BaseFilter):
    name = "🧑‍🍳 Unigram log-prob filter"
    _requires_dependencies = ["nltk"]

    def __init__(
        self,
        logprobs_threshold: float = -10,
        exclusion_writer: DiskWriter = None,
    ):
        """
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold

        Args:
            logprobs_threshold:
            exclusion_writer:
        """
        super().__init__(exclusion_writer)
        self.logprobs_threshold = logprobs_threshold
        self.unigram_frequencies = self.get_frequencies()

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
        from nltk.tokenize import word_tokenize

        words = word_tokenize(doc.text)
        freqs = [
            self.unigram_frequencies.get(word.lower()) for word in words if self.unigram_frequencies.get(word.lower())
        ]
        return sum([np.log(f) for f in freqs]) / len(freqs)

    def filter(self, doc: Document) -> bool:
        """

        :param doc: document
        :return: is_filter
        """

        return self.get_logprob(doc) > self.logprobs_threshold
