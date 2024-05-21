import shutil
import tempfile
import unittest

from datatrove.data import Document
from datatrove.pipeline.dedup.bloom_filter import BloomFilterConfig, SingleBloomFilter
from tests.utils import use_hash_configs


TEXT_0 = (
    "A SQUAT grey building of only thirty-four stories. Over the main entrance the words, CENTRAL LONDON HATCHERY "
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

TEXT_2 = (
    "Arise, arise, Riders of Théoden! Fell deeds awake: fire and slaughter! Spear shall be shaken, shield be "
    "splintered, a sword-day, a red day, ere the sun rises!"
)

TEXT_3 = (
    "I hope you're pleased with yourselves. We could all have been killed — or worse, expelled. Now if you don't "
    "mind, I'm going to bed."
)

TEXT_4 = (
    "Meycauayan Tree is one of the three acacia trees (Samanea saman) located in the patio of the Parish Church "
    "of St. Francis of Assisi in Meycauayan City, Bulacan, Philippines. Planted by an unknown person, it has "
    "stood on the grounds of the parish church for almost a century and a half."
)

TEXT_5 = (
    "Geologically the Castelltallat range is made up of limestone and marl mountains oriented WSW-ENE. "
    "The highest point of the range is the 936 m high 'Tossal'. The northern slopes are steep and forested, "
    "while the southern slopes are used for agriculture owing to their lesser inclination. Most of the mountain "
    "belongs to the municipality of Sant Mateu de Bages while the western part lies within the municipalities "
    "of Pinós and La Molsosa. The village of Castelltallat was a municipality until 1840 when it became part "
    "of San Mateu de Bages municipal term. The parish church of Sant Miquel has been documented since 1031 "
    "and is located at an altitude of 887 m."
)

TEXT_6 = (
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

TEXT_7 = "1 + 1 = 2, 2 + 2 = 4, 4 + 4 = 8, ..."

DOCS = [
    Document(text=TEXT_0, id="0"),
    Document(text=TEXT_1, id="1"),
    Document(text=TEXT_2, id="2"),
    Document(text=TEXT_3, id="3"),
    Document(text=TEXT_4, id="4"),
    Document(text=TEXT_5, id="5"),
    Document(text=TEXT_6, id="6"),
    Document(text=TEXT_7, id="7"),
    Document(text=TEXT_0, id="8"),
    Document(text=TEXT_1, id="9"),
    Document(text=TEXT_6[:-10], id="10"),
]

TARGETS = [True] * 8 + [False] * 3


class BloomFilter(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    @use_hash_configs(precision=[32])
    def test_sd(self, hash_config):
        bloom_filter = SingleBloomFilter(
            output_folder=self.tmp_dir,
            config=BloomFilterConfig(m_bytes=2**10 - 1, k=7, expected_elements=866, hash_config=hash_config),
        )

        for doc_idx, doc in enumerate(DOCS):
            is_unique = bloom_filter.step(doc)
            self.assertEqual(is_unique, TARGETS[doc_idx])

        # # Uncomment to debug:
        # fp = get_false_positive_prob(bloom_filter.m_bytes, n=bloom_filter.total_shingles, k=bloom_filter.k)
        # print(f"False probability = {fp:.3}")
        # print(f"Optimal K given total shingles = {get_optimal_k(bloom_filter.m_bytes, bloom_filter.total_shingles)}")
        # print(f"{bloom_filter.total_shingles=}")
