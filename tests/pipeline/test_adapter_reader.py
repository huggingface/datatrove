import unittest

from datatrove.pipeline.readers import HuggingFaceDatasetReader

from ..utils import require_datasets


@require_datasets
class TestAdapterReader(unittest.TestCase):
    def test_adapter_reader(self):
        def custom_adapter(self, data, path, id_in_file):
            return {
                "text": data[self.text_key] + "\n" + data["best_answer"],  # Example usage of self to access text_key
                "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            }

        reader = HuggingFaceDatasetReader(
            "truthful_qa",
            dataset_options={"name": "generation", "split": "validation"},
            text_key="question",
            id_key="",
            adapter=custom_adapter,
        )
        data = list(reader())
        assert len(data[0].text) == 104
