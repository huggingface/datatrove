"""Tests for media writer and reader integration"""

import shutil
import tempfile
import unittest
from copy import deepcopy

from datatrove.data import Document, Media, MediaType
from datatrove.pipeline.media.media_readers.zstd import ZstdReader
from datatrove.pipeline.media.media_writers.zstd import ZstdWriter


class TestMediaWriterReader(unittest.TestCase):
    """Test media writer and reader integration"""

    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_binary_zstd_writer_reader_cycle(self):
        """Test writing and reading media with BinaryZstdWriter and ZstdThreadedReader"""
        # Create test documents with media
        test_bytes_1 = b"First test data" * 100  # Make it larger for compression
        test_bytes_2 = b"Second test data" * 100
        test_bytes_3 = b"Third test data" * 100

        docs = [
            Document(
                text="Document 1",
                id="doc1",
                media=[
                    Media(
                        id="media1",
                        type=MediaType.IMAGE,
                        url="https://example.com/image1.jpg",
                        media_bytes=test_bytes_1,
                    )
                ],
            ),
            Document(
                text="Document 2",
                id="doc2",
                media=[
                    Media(
                        id="media2",
                        type=MediaType.IMAGE,
                        url="https://example.com/image2.jpg",
                        media_bytes=test_bytes_2,
                    ),
                    Media(
                        id="media3",
                        type=MediaType.VIDEO,
                        url="https://example.com/video.mp4",
                        media_bytes=test_bytes_3,
                    ),
                ],
            ),
        ]

        # Write media
        writer = ZstdWriter(output_folder=self.tmp_dir, compression_level=3)
        written_docs = deepcopy(list(writer.run(iter(docs), rank=0, world_size=1)))

        # Verify media fields are populated
        self.assertEqual(len(written_docs), 2)

        # Read media back
        reader = ZstdReader(data_folder=self.tmp_dir, workers=2)

        # Null the media bytes to simulate a reader that doesn't have the media bytes
        for doc in docs:
            for media in doc.media:
                media.media_bytes = None

        read_docs = list(reader.run(iter(docs), rank=0, world_size=1))

        # Verify
        self.assertEqual(len(read_docs), 2)

        # Sort the read_docs by id as the order is not guaranteed
        read_docs.sort(key=lambda x: x.id)
        written_docs.sort(key=lambda x: x.id)

        # Check that the read_docs are the same as the written_docs
        for read_doc, written_doc in zip(read_docs, written_docs):
            for i in range(len(read_doc.media)):
                self.assertEqual(read_doc.media[i].media_bytes, written_doc.media[i].media_bytes)

    def test_binary_zstd_writer_preserves_order(self):
        """Test that media is written in order"""
        # Create documents with different sized media
        docs = []
        expected_bytes = []
        for i in range(10):
            test_bytes = f"Test data {i}".encode() * 50
            expected_bytes.append(test_bytes)
            docs.append(
                Document(
                    text=f"Document {i}",
                    id=f"doc{i}",
                    media=[
                        Media(
                            id=f"media{i}",
                            type=MediaType.DOCUMENT,
                            url=f"https://example.com/doc{i}.pdf",
                            media_bytes=test_bytes,
                        )
                    ],
                )
            )

        # Write
        writer = ZstdWriter(output_folder=self.tmp_dir)
        written_docs = deepcopy(list(writer.run(iter(docs), rank=0, world_size=1)))

        # Null the media bytes to simulate a reader that doesn't have the media bytes
        for doc in docs:
            for media in doc.media:
                media.media_bytes = None

        # Read with preserve_order=True
        reader = ZstdReader(data_folder=self.tmp_dir, workers=3, preserve_order=True)
        read_docs = list(reader.run(iter(docs), rank=0, world_size=1))

        # Verify order is preserved
        self.assertEqual(len(read_docs), 10)
        for read_doc, written_doc in zip(read_docs, written_docs):
            for i in range(len(read_doc.media)):
                self.assertEqual(read_doc.media[i].id, written_doc.media[i].id)
                self.assertEqual(read_doc.media[i].media_bytes, written_doc.media[i].media_bytes)
