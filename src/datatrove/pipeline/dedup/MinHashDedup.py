from datasketch import MinHash

from datatrove.data import DocumentsPipeline, Document
from transformers import AutoTokenizer
from datatrove.pipeline.base import PipelineStep
import numpy as np
import re


# here we implement lsh minhash, reference: On the resemblance and containment of documents, Broader
# https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf
# A more gentle explanation: http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf


def simplify_content(text):
    # lower cased
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", text.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    text = re.sub(r"\s+", " ", text.strip())
    return text


class CreateSignatures(SomeSortOfWriter):

    def __init__(
            self,
            n_grams: int = 5,
            buckets: int = 450,
            hashes_per_bucket: int = 20,
            *args,
            **kwargs
    ):
        super().__init__(save_filename='&{shard_id}.dedupsig', *args, **kwargs)
        self.n_grams = n_grams
        self.buckets = buckets
        self.per_bucket = hashes_per_bucket
        self.total_hashes = buckets * hashes_per_bucket
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def get_hashes(self, doc: Document):
        content = doc.content
        tokens: list = self.tokenizer(simplify_content(content))
        nr_tokens = len(tokens)
        if nr_tokens < self.n_grams:
            return
        m = MinHash(num_perm=self.total_hashes)
        n_grams = np.array([[hash(tokens[x:x + self.n_grams])] for x in range(len(tokens) - self.n_grams + 1)])
        [m.update(x) for x in n_grams]
        return m.hashvalues

    def writingpart(self):
        pass


class MinHashDedup(PipelineStep):
    def __init__(
            self,
            n_grams: int = 5,
            n_buckets: int = 450,
            hashes_per_bucket: int = 20,

    ):
        """
        Abstract Filter pipeline step.
        A Filter drops samples

        @param args:
        @param kwargs:
        """
        super().__init__()

        self.create_signatures = CreateSignatures(n_grams=n_grams,
                                                  n_buckets=n_buckets,
                                                  hashes_per_bucket=hashes_per_bucket)

    def __call__(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        step method for Filters.
        Drops documents that if .filter() is False

        @param datapipe: input DocumentsPipeline
        @return: DocumentsPipeline
        """

        self.create_signatures()

