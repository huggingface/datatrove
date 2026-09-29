import tempfile
import unittest

from datatrove.pipeline.readers import HuggingFaceDatasetReader

from ...utils import require_datasets


@require_datasets
class TestHuggingFaceReader(unittest.TestCase):
    def test_read_dataset(self):
        reader = HuggingFaceDatasetReader(
            "truthfulqa/truthful_qa",
            dataset_options={"name": "generation", "split": "validation"},
            text_key="question",
        )
        data = list(reader())
        self.assertEqual(len(data), 817)

    def test_read_dataset_shuffle(self):
        reader = HuggingFaceDatasetReader(
            "truthfulqa/truthful_qa",
            dataset_options={"name": "generation", "split": "validation"},
            text_key="question",
            shuffle_files=True,
        )
        data = list(reader())
        self.assertEqual(len(data[0].text), 69)
        self.assertEqual(len(data[1].text), 46)

    def test_read_streaming_dataset(self):
        reader = HuggingFaceDatasetReader(
            "truthfulqa/truthful_qa",
            dataset_options={"name": "generation", "split": "validation"},
            text_key="question",
            streaming=True,
        )
        data = list(reader())
        self.assertEqual(len(data), 817)

    def test_read_streaming_dataset_shuffle(self):
        reader = HuggingFaceDatasetReader(
            "truthfulqa/truthful_qa",
            dataset_options={"name": "generation", "split": "validation"},
            text_key="question",
            streaming=True,
            shuffle_files=True,
        )
        data = list(reader())
        self.assertEqual(len(data[0].text), 69)
        self.assertEqual(len(data[1].text), 46)

    def test_sharding(self):
        for shards in [1, 3]:
            for streaming in [True, False]:
                reader = HuggingFaceDatasetReader(
                    "huggingface/datatrove-tests",
                    dataset_options={"name": f"sharding-{shards}", "split": "train"},
                    text_key="text",
                    streaming=streaming,
                )
                # For streaming == True and sharding-3, the data is not contiguous
                # File1 -> ["hello", "world"], File2 -> ["how", "are"], File3 -> ["you"]
                # Because the data are taken non-contignous first shard gets File1 + File3
                # and second shard gets File2
                data0 = list(reader(rank=0, world_size=2))
                data1 = list(reader(rank=1, world_size=2))
                self.assertEqual(len(data0), 3)
                self.assertEqual(len(data1), 2)


@require_datasets
class TestHuggingFaceReaderSkipLimit(unittest.TestCase):
    def setUp(self):
        from datasets import Dataset

        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        # an empty row checks that skip and limit count only rows with text
        texts = ["doc 0", "doc 1", "", "doc 3", "doc 4", "doc 5", "doc 6", "doc 7"]
        Dataset.from_dict({"text": texts, "id": [str(i) for i in range(len(texts))]}).save_to_disk(self.folder.name)

    def read_ids(self, rank=0, world_size=1, **kwargs):
        reader = HuggingFaceDatasetReader(self.folder.name, load_from_disk=True, **kwargs)
        return [doc.id for doc in reader(rank=rank, world_size=world_size)]

    def test_skip_and_limit(self):
        self.assertEqual(self.read_ids(skip=2, limit=3), ["3", "4", "5"])

    def test_skip_and_limit_apply_per_task(self):
        for rank in range(2):
            rank_ids = self.read_ids(rank=rank, world_size=2)
            self.assertEqual(self.read_ids(rank=rank, world_size=2, skip=1, limit=2), rank_ids[1:3])

    def test_generated_ids_do_not_depend_on_skip(self):
        from datasets import Dataset

        # no id column, so ids are generated; small batches and an empty row around the skip point
        with tempfile.TemporaryDirectory() as folder:
            Dataset.from_dict({"text": ["a", "b", "", "c", "d", "e"]}).save_to_disk(folder)
            read_ids = lambda **kwargs: [  # noqa: E731
                doc.id for doc in HuggingFaceDatasetReader(folder, load_from_disk=True, batch_size=2, **kwargs)()
            ]
            all_ids = read_ids()
            self.assertEqual(len(all_ids), 5)
            for skip in range(7):
                self.assertEqual(read_ids(skip=skip), all_ids[skip:])
            self.assertEqual(read_ids(limit=0), [])
