"""Tests for Media serialization/deserialization in JSONL reader/writer"""

import shutil
import tempfile
import unittest
from copy import deepcopy

from datatrove.data import Document, Media, MediaType
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


class TestMediaSerialization(unittest.TestCase):
    """Test that Media objects with bytes are correctly serialized and deserialized"""

    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_media_bytes_serialization(self):
        """Test that media_bytes are correctly encoded/decoded with base64"""
        # Create test documents with media containing bytes
        test_bytes = b"Hello, World! This is test binary data \x00\x01\x02\xff"
        docs = [
            Document(
                text="Document with image",
                id="doc1",
                media=[
                    Media(
                        id="media1",
                        type=MediaType.IMAGE,
                        url="https://example.com/image.jpg",
                        media_bytes=test_bytes,
                        metadata={"width": 100, "height": 200},
                    )
                ],
                metadata={"source": "test"},
            ),
            Document(
                text="Document with multiple media",
                id="doc2",
                media=[
                    Media(
                        id="media2",
                        type=MediaType.VIDEO,
                        url="https://example.com/video.mp4",
                        media_bytes=b"video data",
                        alt="Test video",
                    ),
                    Media(
                        id="media3",
                        type=MediaType.AUDIO,
                        url="https://example.com/audio.mp3",
                        media_bytes=b"audio data",
                    ),
                ],
                metadata={"source": "test2"},
            ),
            Document(text="Document without media", id="doc3", metadata={"source": "test3"}),
        ]

        # Write documents
        writer = JsonlWriter(output_folder=self.tmp_dir, compression=None, save_media_bytes=True)
        written_docs = deepcopy(list(writer.run(iter(docs), rank=0, world_size=1)))

        # Null the media bytes to simulate a reader that doesn't have the media bytes
        for doc in docs:
            for media in doc.media:
                media.media_bytes = None

        # Read documents back
        reader = JsonlReader(data_folder=self.tmp_dir, compression=None)
        read_docs = list(reader.run(rank=0, world_size=1))

        for read_doc, written_doc in zip(read_docs, written_docs):
            for i in range(len(read_doc.media)):
                self.assertEqual(read_doc.media[i].media_bytes, written_doc.media[i].media_bytes)
