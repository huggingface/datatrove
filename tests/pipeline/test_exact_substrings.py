import os
import shutil
import tempfile
import unittest

from datatrove.data import Document
from datatrove.io import LocalInputDataFolder, LocalOutputDataFolder
from datatrove.pipeline.dedup.exact_substrings import (
    DatasetToSequence,
    DedupReader,
    MergeSequences,
    read_bytes,
    sequence_reader,
)
from datatrove.pipeline.dedup.utils import ExtensionHelperES


TEXT_1 = (
    "Nuclear fusion is a reaction in which two or more atomic nuclei, usually deuterium and tritium "
    "(hydrogen variants), are combined to form one or more different atomic nuclei and subatomic particles "
    "(neutrons or protons). The difference in mass between the reactants and products is manifested as either "
    "the release or absorption of energy."
)

TEXT_2 = (
    "The Adventures of Peregrine Pickle is a picaresque novel by the Scottish author Tobias Smollett, first "
    "published in 1751 and revised and published again in 1758. It tells the story of an egotistical man who "
    "experiences luck and misfortunes in the height of 18th-century European society."
)

TEXT_3 = "A, B, C, D" * 3 + TEXT_2

TEXT_4 = "Apple, Frog, SUN" * 20 + TEXT_2 + "FIRE" * 10

TEXT_5 = (
    "This is but one truly serious philosophical problem, and that is suicide. Judging whether life is or is not worth "
    "living amounts to answering the fundamental question of philosophy. All the rest— whether or not the world has "
    "three dimensions, whether the mind has nine or twelve categories—comes afterwards. These are games; one must "
    "first answer. And if it is true, as Nietzsche claims, that a philosopher, to deserve our respect, must preach "
    "by example, you can appreciate the importance of that reply, for it will precede the definitive act. These are "
    "facts the heart can feel; yet they call for careful study before they become clear to the intellect. If I ask "
    "myself how to judge that this question is more urgent than that, I reply that one judges by the actions it "
    "entails. I have never seen anyone die for the ontological argument. Galileo, who held a scientific truth of "
    "great importance, abjured it with the greatest ease as soon as it endangered his life. In a certain sense, "
    "he did right."
)

TEXT_6 = (
    "I do believe there is only one real philosophical issue, to live or to not. Judging whether life is or is "
    "not worth living amounts to answering the fundamental question of philosophy. All the rest— whether or not "
    "the world has three dimensions, whether the mind has nine or twelve categories—comes afterwards. That is "
    "why I decided to buy a coca zero, no sugar!"
)

TEXT_7 = "A, B, C, D" * 3 + TEXT_1

TEXT_8 = (
    "333 South Wabash is a simple, rectangular International Style building, but it is unique in that the entire "
    "building was painted bright red by Eagle Painting & Maintenance Company, Inc., turning an otherwise "
    "ordinary-looking structure into one of the most eye-catching buildings in the city. It was designed by the "
    "firm of Graham, Anderson, Probst & White and was completed in 1972."
)

TEXT_9 = (
    "The guaiabero (Bolbopsittacus lunulatus) is a species of parrot in the family Psittaculidae, belonging to the "
    "monotypic genus Bolbopsittacus[2] and closely related to the lovebirds (Agapornis) and hanging parrots "
    "(Loriculus).[3] It is endemic to the Philippines and locally known as bubutok. Its common name is derived "
    "from its reputation for eating guavas."
)

TEXT_10 = (
    "The Lake of the Woods Ranger Station is a United States Forest Service compound consisting of eight "
    "buildings overlooking Lake of the Woods in the Fremont-Winema National Forests of southern Oregon. All "
    "of the ranger station structures were built by the Civilian Conservation Corps between 1937 and 1939. "
    "Today, the compound serves as a Forest Service work center, and the old ranger station office is a visitor "
    "center. The ranger station is listed on the National Register of Historic Places."
)

TEXT_11 = (
    "In 1898, the Cascade Forest Reserve was created in southern Oregon. The forest area around Lake of the Woods "
    "became part of the reserve. The forest was managed as part of that Cascade reserve until 1908, when the "
    "area was transferred to the Crater National Forest. The Forest Service issued the first residential use "
    "permit for a private cabin in the Lake of the Woods area in 1916. In 1920, the Forest Service built a "
    "public campground at the lake. The site was very popular, with 1,850 summer visitors using the campground "
    "the first year it open."
)

TEXT_12 = TEXT_11

TEXT_13 = (
    "Solo et pensoso i più deserti campi vo mesurando a passi tardi et lenti, et gli occhi porto per "
    "fuggire intenti ove vestigio human l’arena stampi."
)

bytearange_file = (
    "S 388\nGot 2\nS 364\nS 438\nGot 2\nS 358\nGot 2\nS 298\nGot 2\nS 412\nGot 2\nGot 2\nS 388\nGot 2\n"
    "S 350\nGot 2\nMerging.\nout\n14 158\n168 302\n303 348\n350 484\n496 768\n770 942\n980 1068"
    "\n1406 1494\n1527 1572\n1574 1718\n2291 2526\n2527 2762\n"
)


class TestExactSubstr(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        with open(self.test_dir + "/test" + ExtensionHelperES.stage_3_bytes_ranges, "w") as f:
            f.write(bytearange_file)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_signature(self):
        dataset_to_sequence = DatasetToSequence(output_folder=LocalOutputDataFolder(path=self.test_dir))
        merge_sequence = MergeSequences(
            input_folder=LocalInputDataFolder(path=self.test_dir),
            output_folder=LocalOutputDataFolder(path=self.test_dir),
            tasks_stage_1=1,
        )

        dedup_reader = DedupReader(
            data_folder=LocalInputDataFolder(path=self.test_dir),
            sequence_folder=LocalInputDataFolder(path=self.test_dir),
            min_doc_words=0,
        )

        data = [
            Document(content=TEXT_1, data_id="0"),
            Document(content=TEXT_2, data_id="1"),
            Document(content=TEXT_3, data_id="2"),
            Document(content=TEXT_4, data_id="3"),
            Document(content=TEXT_5, data_id="4"),
            Document(content=TEXT_6, data_id="5"),
            Document(content=TEXT_7, data_id="6"),
            Document(content=TEXT_8, data_id="7"),
            Document(content=TEXT_9, data_id="8"),
            Document(content=TEXT_10, data_id="9"),
            Document(content=TEXT_11, data_id="10"),
            Document(content=TEXT_12, data_id="11"),
            Document(content=TEXT_13, data_id="12"),
        ]

        # test quality of stage 1, 2 output
        dataset_to_sequence(data=data)
        merge_sequence(data=[])
        big_sequence_path = self.test_dir + "/dataset" + ExtensionHelperES.stage_2_big_sequence
        self.assertTrue(os.path.isfile(big_sequence_path))

        dedup_reader.rank = 0
        dedup_reader.read_bytes_offset()
        bytes_offset = dedup_reader.bytes_offset
        with open(os.path.join(self.test_dir, "00000.es_sequence"), "rb") as f_s:
            with open(os.path.join(self.test_dir, "dataset.big_sequence"), "rb") as f_b:
                sequence = f_s.read()
                self.assertEqual(sequence, f_b.read())
                self.assertEqual(len(sequence), bytes_offset[1])

        sequence_file, size_file = dedup_reader.get_all_files(0, 1)
        for i, doc_content in enumerate(sequence_reader(sequence_file, size_file)):
            self.assertEqual(data[i].content, dedup_reader.tokenizer.decode(read_bytes(doc_content)))

        # test if  deduplication actually works
        for i, doc in enumerate(dedup_reader(data=data)):
            if i == 0:
                self.assertEqual(doc.content, "N")

            if i == 1:
                self.assertEqual(doc.content, "")

            if i == 2:
                self.assertEqual(doc.content, " D")

            if i == 3:
                self.assertEqual(doc.content, "Apple SUN")

            if i == 4:
                self.assertEqual(doc.content, TEXT_5[:72] + TEXT_5[307:])

            if i == 5:
                self.assertEqual(doc.content, TEXT_6[:74] + TEXT_6[309:])

            if i == 6:
                self.assertEqual(doc.content, " DN")

            if i == 7:
                self.assertEqual(doc.content, TEXT_8)

            if i == 8:
                self.assertEqual(doc.content, TEXT_9)

            if i == 9:
                self.assertEqual(doc.content, TEXT_10)

            if i == 10:
                self.assertEqual(doc.content, "")

            if i == 11:
                self.assertEqual(doc.content, TEXT_12)

            if i == 12:
                self.assertEqual(doc.content, TEXT_13)
