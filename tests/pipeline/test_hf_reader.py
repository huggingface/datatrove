import unittest

from datatrove.pipeline.readers import HuggingFaceDatasetReader

from ..utils import require_datasets


@require_datasets
class TestHuggingFaceReader(unittest.TestCase):
    def test_read_dataset(self):
        reader = HuggingFaceDatasetReader(
            "truthful_qa", dataset_options={"name": "generation", "split": "validation"}, text_key="question"
        )
        data = list(reader())
        self.assertEqual(len(data), 817)

    def test_read_streaming_dataset(self):
        reader = HuggingFaceDatasetReader(
            "truthful_qa",
            dataset_options={"name": "generation", "split": "validation"},
            text_key="question",
            streaming=True,
        )
        data = list(reader())
        self.assertEqual(len(data), 817)

    def test_sharding(self):
        reader = HuggingFaceDatasetReader(
            "truthful_qa",
            dataset_options={"name": "generation", "split": "validation"},
            text_key="question",
            streaming=True,
        )
        data0 = list(reader(rank=0, world_size=2))
        data1 = list(reader(rank=1, world_size=2))

        self.assertEqual(len(data0), 409)
        self.assertEqual(len(data1), 408)
