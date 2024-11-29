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

    def test_read_dataset_shuffle(self):
        reader = HuggingFaceDatasetReader(
            "truthful_qa",
            dataset_options={"name": "generation", "split": "validation"},
            text_key="question",
            shuffle_files=True,
        )
        data = list(reader())
        self.assertEqual(len(data[0].text), 69)
        self.assertEqual(len(data[1].text), 46)

    def test_read_streaming_dataset(self):
        reader = HuggingFaceDatasetReader(
            "truthful_qa",
            dataset_options={"name": "generation", "split": "validation"},
            text_key="question",
            streaming=True,
        )
        data = list(reader())
        self.assertEqual(len(data), 817)

    def test_read_streaming_dataset_shuffle(self):
        reader = HuggingFaceDatasetReader(
            "truthful_qa",
            dataset_options={"name": "generation", "split": "validation"},
            text_key="question",
            streaming=True,
            shuffle_files=True,
        )
        data = list(reader())
        self.assertEqual(len(data[0].text), 69)
        self.assertEqual(len(data[1].text), 46)

    def test_sharding_1(self):
        """
        >>> ds = load_dataset("huggingface/datatrove-tests",name="sharding-1",split="train",streaming=True)
        >>> ds
        IterableDataset({
            features: ['text'],
            num_shards: 1
        })
        
        >>> print(list(ds.shard(num_shards=2, index=0)))
        [{'text': 'hello'}, {'text': 'world'}, {'text': 'how'}, {'text': 'are'}, {'text': 'you'}]

        >>> print(list(ds.shard(num_shards=2, index=1)))
        IndexError: list index out of range
        
        >>> ds = load_dataset("huggingface/datatrove-tests",name="sharding-1",split="train",streaming=False)
        >>> ds
        Dataset({
            features: ['text'],
            num_rows: 5
        })
        
        >>> print(list(ds.shard(num_shards=2, index=0)))
        >>> print(list(ds.shard(num_shards=2, index=1)))
        [{'text': 'hello'}, {'text': 'world'}, {'text': 'how'}]
        [{'text': 'are'}, {'text': 'you'}]
        
        >>> print(list(ds.shard(num_shards=3, index=0)))
        >>> print(list(ds.shard(num_shards=3, index=1)))
        >>> print(list(ds.shard(num_shards=3, index=2)))
        [{'text': 'hello'}, {'text': 'world'}]
        [{'text': 'how'}, {'text': 'are'}]
        [{'text': 'you'}]

        """
        for streaming in [True, False]:
            reader = HuggingFaceDatasetReader(
                "huggingface/datatrove-tests",
                dataset_options={"name": f"sharding-1", "split": "train"},
                text_key="text",
                streaming=streaming,
            )
            data0 = list(reader(rank=0, world_size=2))
            data1 = list(reader(rank=1, world_size=2))

            self.assertEqual(len(data0), 3)
            self.assertEqual(len(data1), 2)
            
    def test_sharding_3_stream(self):
        """
        >>> ds_stream = load_dataset("huggingface/datatrove-tests",name="sharding-3",split="train",streaming=True)
        >>> ds_stream
        IterableDataset({
            features: ['text'],
            num_shards: 3
        })
        
        >>> print(list(ds_stream.shard(num_shards=2, index=0)))
        >>> print(list(ds_stream.shard(num_shards=2, index=1)))
        [{'text': 'hello'}, {'text': 'world'}, {'text': 'how'}, {'text': 'are'}]
        [{'text': 'you'}]
        
        >>> print(list(list(ds_stream.shard(num_shards=3, index=0))))
        >>> print(list(list(ds_stream.shard(num_shards=3, index=1))))
        >>> print(list(list(ds_stream.shard(num_shards=3, index=2))))
        [{'text': 'hello'}, {'text': 'world'}]
        [{'text': 'how'}, {'text': 'are'}]
        [{'text': 'you'}]
        
        """
        reader = HuggingFaceDatasetReader(
            "huggingface/datatrove-tests",
            dataset_options={"name": f"sharding-3", "split": "train"},
            text_key="text",
            streaming=True,
        )
        data0 = list(reader(rank=0, world_size=2))
        data1 = list(reader(rank=1, world_size=2))

        self.assertEqual(len(data0), 4)
        self.assertEqual(len(data1), 1)
 
    def test_sharding_3(self):
        """
        >>> ds = load_dataset("huggingface/datatrove-tests",name="sharding-3",split="train",streaming=False)
        >>> ds
        Dataset({
            features: ['text'],
            num_rows: 5
        })
        
        >>> print(list(ds.shard(num_shards=2, index=0)))
        >>> print(list(ds.shard(num_shards=2, index=1)))
        [{'text': 'hello'}, {'text': 'world'}, {'text': 'how'}]
        [{'text': 'are'}, {'text': 'you'}]
        
        >>> print(list(ds.shard(num_shards=3, index=0)))
        >>> print(list(ds.shard(num_shards=3, index=1)))
        >>> print(list(ds.shard(num_shards=3, index=2)))
        [{'text': 'hello'}, {'text': 'world'}]
        [{'text': 'how'}, {'text': 'are'}]
        [{'text': 'you'}]
        
        """
        reader = HuggingFaceDatasetReader(
            "huggingface/datatrove-tests",
            dataset_options={"name": f"sharding-3", "split": "train"},
            text_key="text",
            streaming=False,
        )
        data0 = list(reader(rank=0, world_size=2))
        data1 = list(reader(rank=1, world_size=2))

        self.assertEqual(len(data0), 3)
        self.assertEqual(len(data1), 2)
 
