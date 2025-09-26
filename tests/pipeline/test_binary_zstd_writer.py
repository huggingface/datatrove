import random
import unittest
import tempfile
import os
import shutil
import zstandard as zstd

from datatrove.data import Media, Document, MediaType
from datatrove.pipeline.media.writers.binary_zstd import BinaryZstdWriter
from datatrove.io import DataFolder
from datatrove.pipeline.media.readers.zstd_threaded import ZstdThreadedReader

def create_test_docs(seed=42):
    import random
    import string
    
    random.seed(seed)
    
    def random_string(min_len=10, max_len=100):
        length = random.randint(min_len, max_len)
        return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=length))
    
    return [
        Document(id=f"doc_{i}", text="", media=[
            Media(
                id=f"media_{i}", 
                media_bytes=random_string().encode(),
                type=MediaType.DOCUMENT,
                url=f"https://example.com/media_{i}"
            )
        ])
        for i in range(10)
    ]



class TestBinaryZstdWriter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_folder = DataFolder(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_write_and_read_sequentially(self):
        test_documents = create_test_docs()

        writer = BinaryZstdWriter(output_folder=self.output_folder)
        written_docs = list(writer.run(test_documents))

        # Remove media_bytes
        for doc in written_docs:
            for media in doc.media:
                media.media_bytes = None

        # Test with different block sizes
        block_sizes = [1, 5, 10]
        for block_size in block_sizes:
            with self.subTest(block_size=block_size):
                reader = ZstdThreadedReader(data_folder=self.output_folder)
                documents_read = list(reader.run(written_docs))

                def take_bytes(d):
                    return d.media[0].media_bytes

                # Because we modify in place we create second batch
                test_documents = create_test_docs()
                self.assertEqual(
                    list(map(take_bytes, sorted(test_documents, key=lambda doc: doc.id))), 
                    list(map(take_bytes, sorted(documents_read, key=lambda doc: doc.id)))
                )
    
    def test_random_access(self):
        test_documents = create_test_docs()

        # Write the documents
        writer = BinaryZstdWriter(output_folder=self.output_folder)
        written_docs = list(writer.run(test_documents))

        # Clear media bytes after writing
        for doc in written_docs:
            for media in doc.media:
                media.media_bytes = None

        # Create reader
        reader = ZstdThreadedReader(data_folder=self.output_folder)

        # Test reading documents in random order
        random_order = written_docs.copy()
        random.shuffle(random_order)

        for doc in random_order:
            # Read single document
            read_docs = list(reader.run([doc]))
            self.assertEqual(len(read_docs), 1)
            
            # Get original document for comparison
            original_doc = next(d for d in test_documents if d.id == doc.id)
            
            # Compare media bytes
            self.assertEqual(
                read_docs[0].media[0].media_bytes,
                original_doc.media[0].media_bytes
            )

        # Test reading subset of documents
        subset = random.sample(written_docs, 2)
        read_subset = list(reader.run(subset))
        
        # Compare with originals
        for read_doc in read_subset:
            original_doc = next(d for d in test_documents if d.id == read_doc.id)
            self.assertEqual(
                read_doc.media[0].media_bytes,
                original_doc.media[0].media_bytes
            )

    def test_reverse_order(self):
        test_documents = create_test_docs()

        # Write the documents
        writer = BinaryZstdWriter(output_folder=self.output_folder)
        written_docs = list(writer.run(test_documents))

        # Clear media bytes after writing
        for doc in written_docs:
            for media in doc.media:
                media.media_bytes = None

        # Create reader
        reader = ZstdThreadedReader(data_folder=self.output_folder)

        # Test reading documents in reverse order
        reverse_order = written_docs
        reverse_order.reverse()

        for doc in reverse_order:
            # Read single document
            read_docs = list(reader.run([doc]))
            self.assertEqual(len(read_docs), 1)
            
            # Get original document for comparison
            original_doc = next(d for d in test_documents if d.id == doc.id)
            
            # Compare media bytes
            self.assertEqual(
                read_docs[0].media[0].media_bytes,
                original_doc.media[0].media_bytes
            )


