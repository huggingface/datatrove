import os
import shutil
import struct
import tempfile
import unittest

import numpy as np
from tokenizers import Tokenizer

from datatrove.data import Document
from datatrove.io import BaseInputDataFolder, LocalInputDataFolder, LocalOutputDataFolder
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from datatrove.tools.check_dataset import check_dataset, load_doc_ends, load_input_mmap


texts = [
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
data = np.array_split([Document(content=text, data_id=id) for id, text in enumerate(texts)], WORKERS)


def get_texts_from_tokens(input_folder: BaseInputDataFolder):
    tokenizer = Tokenizer.from_pretrained(TOKENIZER)
    texts_from_tokens = []
    for tokens_file, index_file in zip(
        input_folder.list_files(extension=".ds"), input_folder.list_files(extension=".ds.index")
    ):
        doc_ends = load_doc_ends(index_file).tolist()
        tokens_bytes = load_input_mmap(tokens_file)
        for start, end in zip([0] + doc_ends[:-1], doc_ends):
            texts_from_tokens.append(
                tokenizer.decode(struct.unpack("<%sH" % (end - start), tokens_bytes[start * 2 : end * 2]))
            )
    return texts_from_tokens


def check_order_reconstruction(input_folder, mapping):
    texts_from_tokens = get_texts_from_tokens(input_folder)
    if not mapping:
        mapping = range(len(texts))
    for map, from_tokens in zip(mapping, texts_from_tokens):
        assert texts[map] == from_tokens


class TestTokenization(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def run_test(self, dir_name, seed=None, dist_mapping=None, merge_mapping=None):
        TOKENS_DIR = os.path.join(self.test_dir, dir_name)
        MERGED_DIR = os.path.join(self.test_dir, dir_name + "_merged")

        document_tokenizer = DocumentTokenizer(LocalOutputDataFolder(TOKENS_DIR), shuffle=seed is not None, seed=seed)
        for worker, worker_data in enumerate(data):
            document_tokenizer(worker_data, rank=worker, world_size=WORKERS)
        # general consistency check
        input_folder = LocalInputDataFolder(TOKENS_DIR)
        check_dataset(input_folder)

        # check order/reconstruction
        check_order_reconstruction(input_folder, dist_mapping)

        # testing merger
        merger = DocumentTokenizerMerger(
            LocalInputDataFolder(TOKENS_DIR),
            LocalOutputDataFolder(MERGED_DIR),
            save_filename="my_dataset",
            shuffle=seed is not None,
            seed=seed,
        )
        merger(None)

        # general consistency check
        input_folder = LocalInputDataFolder(MERGED_DIR)
        check_dataset(input_folder)

        # check order/reconstruction
        check_order_reconstruction(input_folder, merge_mapping)

    def test_tokenizer_unshuffled(self):
        self.run_test("tokenized_unshuffled")

    def test_tokenizer_shuffled(self):
        self.run_test("tokenized_shuffled", 7383, [2, 0, 1, 4, 3, 5, 7, 6], [2, 4, 7, 0, 3, 1, 6, 5])
