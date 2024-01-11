import copy
import random
import shutil
import string
import tempfile
import unittest

from datatrove.data import Document
from datatrove.pipeline.dedup.sentence_dedup import SentenceDedupFilter, SentenceDedupSignature, SentenceFindDedups


def get_random_string(n: int = 20):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n)) + "."


SENTENCE_ = "A SQUAT grey building of only thirty-four stories."
TEXT_0 = (
    SENTENCE_ + " Over the main entrance the words, CENTRAL LONDON HATCHERY "
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

TEXT_1 = (
    "Wintriness responded to wintriness. The overalls of the workers were white, their "
    "hands gloved with a pale corpse-coloured rubber. The light was frozen, dead, a ghost. Only from the yellow "
    "barrels of the microscopes did it borrow a certain rich and living substance, lying along the polished tubes "
    "like butter, streak after luscious streak in long recession down the work tables. What wintriness even mean ?"
    "If you google it you will find that pretty bus it is used as a word in this book and pretty much it."
)

EXPECTED_TEXT_1 = (
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

TEXT_0_1 = (
    "Meycauayan Tree is one of the three acacia trees (Samanea saman) located in the patio of the Parish Church "
    "of St. Francis of Assisi in Meycauayan City, Bulacan, Philippines. Planted by an unknown person, it has "
    "stood on the grounds of the parish church for almost a century and a half."
)

TEXT_1_1 = (
    "Geologically the Castelltallat range is made up of limestone and marl mountains oriented WSW-ENE. "
    "The highest point of the range is the 936 m high 'Tossal'. The northern slopes are steep and forested, "
    "while the southern slopes are used for agriculture owing to their lesser inclination. Most of the mountain "
    "belongs to the municipality of Sant Mateu de Bages while the western part lies within the municipalities "
    "of Pinós and La Molsosa. The village of Castelltallat was a municipality until 1840 when it became part "
    "of San Mateu de Bages municipal term. The parish church of Sant Miquel has been documented since 1031 "
    "and is located at an altitude of 887 m."
)

TEXT_2_1 = (
    "Chukanovite was first discovered in weathered cavities of a meteorite which fell near the small village "
    "of Dronino, 350 km southeast of Moscow, Russia, but the mineral has since been found elsewhere in cavities"
    " of other iron-rich meteorites. It occurs primarily in association with goethite, akaganeite, hematite, "
    "hibbingite, reevesite, honessite, and kamacite, though the meteorites that contain chukanovite also tend "
    "to contain taenite and chromite. Individual crystals form from a reaction between kamacite and cold "
    "water that is rich in dissolved carbon dioxide, during which they adopt a fibrous to acicular habit and "
    "grow to an average size of roughly 0.5 mm in length and 2-3 μm in thickness. Individual crystals tend to "
    "coalesce within the meteorite cavities into porous collections or crusts of spherulites, each with a "
    "diameter of about 1 mm."
)

TEXT_3_1 = "1 + 1 = 2, 2 + 2 = 4, 4 + 4 = 8, ..."

rs_1 = get_random_string()
rs_2 = get_random_string()

DOCS = [
    Document(content=TEXT_0, data_id="0"),
    Document(content=TEXT_1, data_id="1"),
    Document(content=" ".join([TEXT_0, rs_1]), data_id="2"),
    Document(content=" ".join([rs_1, TEXT_0, rs_2]), data_id="3"),
    Document(content=LOTR, data_id="4"),
    Document(content=" ".join([LOTR, HPPS]), data_id="5"),
]

DOCS_2 = [
    Document(content=TEXT_0_1, data_id="0"),
    Document(content=TEXT_1_1, data_id="1"),
    Document(content=TEXT_2_1, data_id="2"),
    Document(content=LOTR, data_id="3"),
    Document(content=TEXT_3_1, data_id="4"),
]

TARGETS = [
    TEXT_0,
    EXPECTED_TEXT_1,
    rs_1,
    " ".join([rs_1, rs_2]),
    LOTR,
    HPPS,
]

TARGETS_WS2_0 = [
    TEXT_0,
    EXPECTED_TEXT_1,
    rs_1,
    " ".join([rs_1, rs_2]),
    HPPS,
]

TARGETS_WS2_1 = [
    TEXT_0_1,
    TEXT_1_1,
    TEXT_2_1,
    LOTR,
    TEXT_3_1,
]


class SentenceDedup(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_sd(self):
        signature_creation = SentenceDedupSignature(output_folder=self.tmp_dir)
        find_duplicates = SentenceFindDedups(data_folder=self.tmp_dir, output_folder=self.tmp_dir)
        dedup_filter = SentenceDedupFilter(data_folder=self.tmp_dir, min_doc_words=0)

        signature_creation(data=DOCS)
        find_duplicates(data=[])
        for i, doc in enumerate(dedup_filter(data=copy.deepcopy(DOCS))):
            self.assertEqual(doc.content, TARGETS[i])

    def test_sd_worker(self):
        signature_creation = SentenceDedupSignature(output_folder=self.tmp_dir)

        find_duplicates = SentenceFindDedups(data_folder=self.tmp_dir, output_folder=self.tmp_dir)
        dedup_filter = SentenceDedupFilter(data_folder=self.tmp_dir, min_doc_words=0)

        signature_creation(data=DOCS, rank=0, world_size=2)
        signature_creation(data=DOCS_2, rank=1, world_size=2)
        find_duplicates(data=[])

        for i, doc in enumerate(dedup_filter(data=copy.deepcopy(DOCS), rank=0, world_size=2)):
            self.assertEqual(doc.content, TARGETS_WS2_0[i], msg=str(i))

        for i, doc in enumerate(dedup_filter(data=copy.deepcopy(DOCS_2), rank=1, world_size=2)):
            self.assertEqual(doc.content, TARGETS_WS2_1[i])
