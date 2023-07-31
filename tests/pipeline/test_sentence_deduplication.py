import random
import shutil
import string
import tempfile
import unittest

from datatrove.data import Document
from datatrove.io import LocalInputDataFolder, LocalOutputDataFolder
from datatrove.pipeline.dedup.sentence_dedup import SentenceDedupFilter, SentenceDedupSignature, SentenceFindDedups


def get_random_string(n: int = 20):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n)) + "."


SENTENCE_1 = "A SQUAT grey building of only thirty-four stories."
TEXT1 = (
    SENTENCE_1 + " Over the main entrance the words, CENTRAL LONDON HATCHERY "
    "AND CONDITIONING CENTRE, and, in a shield, the World State's motto, COMMUNITY, IDENTITY, STABILITY. The enormous"
    " room on the ground floor faced towards the north. Cold for all the summer beyond the panes, for all the "
    "tropical heat of the room itself, a harsh thin light glared through the windows, hungrily seeking some draped "
    "lay figure, some pallid shape of academic goose-flesh, but finding only the glass and nickel and bleakly shining"
    " porcelain of a laboratory. Wintriness responded to wintriness. The overalls of the workers were white, their "
    "hands gloved with a pale corpse-coloured rubber. The light was frozen, dead, a ghost. Only from the yellow "
    "barrels of the microscopes did it borrow a certain rich and living substance, lying along the polished tubes "
    "like butter, streak after luscious streak in long recession down the work tables. And this, said the Director "
    "opening the door, 'is the Fertilizing Room.'"
)

TEXT2 = (
    "Wintriness responded to wintriness. The overalls of the workers were white, their "
    "hands gloved with a pale corpse-coloured rubber. The light was frozen, dead, a ghost. Only from the yellow "
    "barrels of the microscopes did it borrow a certain rich and living substance, lying along the polished tubes "
    "like butter, streak after luscious streak in long recession down the work tables. What wintriness even mean ?"
    "If you google it you will find that pretty bus it is used as a word in this book and pretty much it."
)

EXPECTED_TEXT_2 = (
    "What wintriness even mean ?"
    "If you google it you will find that pretty bus it is used as a word in this book and pretty much it."
)

LOTR = (
    "Arise, arise, Riders of Théoden! Fell deeds awake: fire and slaughter! Spear shall be shaken, shield be "
    "splintered, a sword-day, a red day, ere the sun rises!"
)

HPPS = (
    "I hope you're pleased with yourselves. We could all have been killed — or worse, expelled. Now if you don't "
    "mind, I'm going to bed."
)


class SentenceDedup(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_signature(self):
        signature_creation = SentenceDedupSignature(output_folder=LocalOutputDataFolder(self.test_dir))
        find_duplicates = SentenceFindDedups(
            data_folder=LocalInputDataFolder(self.test_dir), output_folder=LocalOutputDataFolder(self.test_dir)
        )
        dedup_filter = SentenceDedupFilter(data_folder=LocalInputDataFolder(self.test_dir), min_doc_words=0)

        rs_1 = get_random_string()
        rs_2 = get_random_string()

        docs = [
            Document(content=TEXT1, data_id="0"),
            Document(content=TEXT2, data_id="1"),
            Document(content=" ".join([TEXT1 + rs_1]), data_id="3"),
            Document(content=" ".join([rs_1 + TEXT1 + rs_2]), data_id="4"),
            Document(content=LOTR, data_id="5"),
            Document(content=" ".join([LOTR, HPPS]), data_id="6"),
        ]

        signature_creation(data=docs)
        find_duplicates(data=[])
        for i, doc in enumerate(dedup_filter(data=docs)):
            if i == 0:
                self.assertEqual(doc.content, TEXT1)
            if i == 1:
                self.assertEqual(doc.content, EXPECTED_TEXT_2)
            if i == 2:
                self.assertEqual(doc.content, "'" + rs_1)
            if i == 3:
                self.assertEqual(doc.content, " ".join([rs_1 + SENTENCE_1 + " '" + rs_2]))
            if i == 5:
                print(i)
                self.assertEqual(doc.content, HPPS)
