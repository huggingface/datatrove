import os
import shutil
import struct
import tempfile
import unittest

import numpy as np

from datatrove.data import Document
from datatrove.io import DataFolder, get_datafolder
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from datatrove.tools.check_dataset import check_dataset, load_doc_ends
from datatrove.utils._import_utils import is_tokenizers_available

from ..utils import require_tokenizers


if is_tokenizers_available():
    from tokenizers import Tokenizer


TEXTS = [
    "Life, although it may only be an accumulation of anguish, is dear to me, and I will defend it.",
    "I do know that for the sympathy of one living being, I would make peace with all. I have love in me the likes of which you can scarcely imagine and rage the likes of which you would not believe. If I cannot satisfy the one, I will indulge the other.",
    "Even broken in spirit as he is, no one can feel more deeply than he does the beauties of nature. The starry sky, the sea, and every sight afforded by these wonderful regions, seems still to have the power of elevating his soul from earth. Such a man has a double existence: he may suffer misery, and be overwhelmed by disappointments; yet, when he has retired into himself, he will be like a celestial spirit that has a halo around him, within whose circle no grief or folly ventures.",
    "How mutable are our feelings, and how strange is that clinging love we have of life even in the excess of misery!",
    "It is true, we shall be monsters, cut off from all the world; but on that account we shall be more attached to one another.",
    "The fallen angel becomes a malignant devil. Yet even that enemy of God and man had friends and associates in his desolation; I am alone.",
    "With how many things are we on the brink of becoming acquainted, if cowardice or carelessness did not restrain our inquiries."
    "Hateful day when I received life!' I exclaimed in agony. 'Accursed creator! Why did you form a monster so hideous that even you turned from me in disgust? God, in pity, made man beautiful and alluring, after his own image; but my form is a filthy type of yours, more horrid even from the very resemlance. Satan had his companions, fellow-devils, to admire and encourage him; but I am solitary and abhorred.",
    "I looked upon the sea, it was to be my grave",
]

TOKENIZER = "gpt2"
WORKERS = 3
DATA = np.array_split([Document(text=text, id=id) for id, text in enumerate(TEXTS)], WORKERS)


def get_texts_from_tokens(input_folder: DataFolder):
    tokenizer = Tokenizer.from_pretrained(TOKENIZER)
    texts_from_tokens = []
    for tokens_file, index_file in zip(
        input_folder.list_files(glob_pattern="*.ds"), input_folder.list_files(glob_pattern="*.ds.index")
    ):
        doc_ends = load_doc_ends(input_folder.open(index_file, "rb"))
        with input_folder.open(tokens_file, "rb") as f:
            for start, end in zip([0] + doc_ends[:-1], doc_ends):
                texts_from_tokens.append(
                    tokenizer.decode(struct.unpack("<%sH" % (end - start), f.read((end - start) * 2)))
                )
    return texts_from_tokens


@require_tokenizers
class TestTokenization(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def check_order_reconstruction(self, input_folder, mapping):
        texts_from_tokens = get_texts_from_tokens(input_folder)
        if not mapping:
            mapping = range(len(TEXTS))
        for map, from_tokens in zip(mapping, texts_from_tokens):
            self.assertEqual(TEXTS[map], from_tokens)

    def test_tokenizer(self):
        for sub_test, args in [
            ("tokenizer_unshuffled", (None, None, None)),
            ("tokenizer_shuffled", (7383, [2, 0, 1, 4, 3, 5, 7, 6], [2, 4, 7, 0, 3, 1, 6, 5])),
        ]:
            with self.subTest(sub_test):
                seed, dist_mapping, merge_mapping = args

                TOKENS_DIR = os.path.join(self.tmp_dir, sub_test, "tokens")
                MERGED_DIR = os.path.join(self.tmp_dir, sub_test, "merged")

                document_tokenizer = DocumentTokenizer(
                    TOKENS_DIR, local_working_dir=None, shuffle=seed is not None, seed=seed, save_loss_metadata=True
                )
                for worker, worker_data in enumerate(DATA):
                    document_tokenizer(worker_data, rank=worker, world_size=WORKERS)
                # general consistency check
                input_folder = get_datafolder(TOKENS_DIR)
                check_dataset(input_folder)

                # check order/reconstruction
                self.check_order_reconstruction(input_folder, dist_mapping)

                # testing merger
                merger = DocumentTokenizerMerger(
                    TOKENS_DIR,
                    MERGED_DIR,
                    save_filename="my_dataset",
                    shuffle=seed is not None,
                    save_loss_metadata=True,
                    seed=seed,
                )
                merger(None)

                # general consistency check
                input_folder = get_datafolder(MERGED_DIR)
                check_dataset(input_folder)

                # check order/reconstruction
                self.check_order_reconstruction(input_folder, merge_mapping)
