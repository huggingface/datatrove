import os
import urllib.request

import numpy as np
from huggingface_hub import cached_assets_path
from loguru import logger
from nltk.tokenize import word_tokenize

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


PANDAS_INSTALLED = True
try:
    import pandas as pd
except ImportError:
    PANDAS_INSTALLED = False

UNIGRAM_DOWNLOAD = "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/lucas/google-1T-unigram/unigram_freq.csv"


class UnigramLogProbFilter(BaseFilter):
    name = "🧑‍🍳 Unigram log-prob filter"

    def __init__(
        self,
        logprobs_threshold: float = -10,
        exclusion_writer: DiskWriter = None,
    ):
        """
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold

        @param languages: list of languages to not filter out.
        """
        super().__init__(exclusion_writer)
        self.logprobs_threshold = logprobs_threshold
        self.unigram_frequencies = self.get_frequencies()

    def get_frequencies(self):
        if not PANDAS_INSTALLED:
            raise ImportError("Pandas need to be installed to use unigram filter")
        download_dir = cached_assets_path(
            library_name="datatrove", namespace="filters", subfolder="unigram_logprob_filter"
        )
        unigram_freq_file = os.path.join(download_dir, "unigram_freq.csv")
        if not os.path.isfile(unigram_freq_file):
            logger.info("⬇️ Downloading unigram-frequencies ...")
            urllib.request.urlretrieve(UNIGRAM_DOWNLOAD, unigram_freq_file)

        df = pd.read_csv(unigram_freq_file)
        df["count"] = df["count"] / df["count"].sum()
        return dict(zip(df["word"], df["count"]))

    def get_logprob(self, doc):
        words = word_tokenize(doc.content)
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
