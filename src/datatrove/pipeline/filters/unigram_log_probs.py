import os
import urllib.request

import numpy as np
from loguru import logger
from nltk.tokenize import word_tokenize

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.typeshelper import LocalPaths


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
        model_local_path: str = os.path.join(LocalPaths.download, "logprob_filter/"),
        **kwargs,
    ):
        """
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold

        @param languages: list of languages to not filter out.
        """
        super().__init__(**kwargs)
        self.logprobs_threshold = logprobs_threshold
        self.model_local_path = model_local_path
        self.unigram_frequencies = self.get_frequencies()

    def get_frequencies(self):
        if not PANDAS_INSTALLED:
            raise ImportError("Pandas need to be installed to use unigram filter")
        if not os.path.isfile(self.model_local_path + "unigram_freq.csv"):
            os.makedirs(os.path.dirname(self.model_local_path), exist_ok=True)
            logger.info("⬇️ Downloading unigram-frequencies ...")
            urllib.request.urlretrieve(UNIGRAM_DOWNLOAD, self.model_local_path + "unigram_freq.csv")

        df = pd.read_csv(self.model_local_path + "unigram_freq.csv")
        df["count"] = df["count"] / df["count"].sum()
        return dict(zip(df["word"], df["count"]))

    def filter(self, doc: Document) -> bool:
        """

        :param doc: document
        :return: is_filter
        """

        words = word_tokenize(doc.content)
        freqs = [
            self.unigram_frequencies.get(word.lower()) for word in words if self.unigram_frequencies.get(word.lower())
        ]
        logprob = sum([np.log(f) for f in freqs]) / len(freqs)
        return logprob > self.logprobs_threshold
