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
        assert len(data) == 817
