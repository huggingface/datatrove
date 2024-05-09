import copy
import shutil
import tempfile
import unittest

from datatrove.data import Document
from datatrove.pipeline.decont import NGramsDecontConfig, NGramsDecontFilter, NGramsDecontIndexer
from tests.utils import require_xxhash, use_hash_configs


TEXTS = [
    "A lady walks to a barbell. She bends down and grabs the pole.",  # 0: contaminated query
    "get into formation, then begin dancing and flipping as male cheerleaders join them.",  # 1: contaminated label
    "He is using commercial lawn mowing equipment. he walks back and forth as he mows the grass.",  # 2: cont overlap
    "He is using commercial lawn mowing equipment. he is animated as he does the task.",  # 3: incorrect completion
    "walks outside plugs his lawn mower in and gets ready to mow",  # 4: single contaminated query ngram
    "",  # 5: not contaminated at all
    "walks outside plugs his lawn mower in and gets ready to",  # 6: single contaminated query text < 1 ngram
]

DOCS = [
    Document(
        text="Nothing is so painful to the human mind as a great and sudden change. "
        + text
        + " Beware; for I am fearless, and therefore powerful.",
        id=str(text_i),
    )
    for text_i, text in enumerate(TEXTS)
]


@require_xxhash
class TestNGramDecont(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def get_test_results(self, config):
        indexer = NGramsDecontIndexer(self.tmp_dir, lighteval_tasks="leaderboard|hellaswag", config=config)
        indexer.run()
        nfilter = NGramsDecontFilter(self.tmp_dir, config=config)
        return tuple([int(doc.id) for doc in nfilter(copy.deepcopy(DOCS))])

    @use_hash_configs()
    def test_label_only(self, hash_config):
        self.assertEqual(
            self.get_test_results(
                NGramsDecontConfig(find_query_ngrams=False, find_overlap_ngrams=False, hash_config=hash_config)
            ),
            (0, 2, 3, 4, 5, 6),
        )

    def test_query(self):
        self.assertEqual(
            self.get_test_results(NGramsDecontConfig(find_query_ngrams=True, find_overlap_ngrams=False)), (2, 3, 5, 6)
        )

    def test_overlap(self):
        self.assertEqual(
            self.get_test_results(NGramsDecontConfig(find_query_ngrams=False, find_overlap_ngrams=True)),
            (0, 3, 4, 5, 6),
        )
