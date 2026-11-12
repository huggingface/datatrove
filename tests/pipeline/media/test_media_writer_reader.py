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

    def test_binary_zstd_writer_reader_filename_tags(self) -> None:
        """Round-trip media stored using its own ID and metadata in the filename."""
        media = [
            Media(
                id="image1",
                type=MediaType.IMAGE,
                url="https://example.com/image.jpg",
                media_bytes=b"image data\x00\xff",
                metadata={"collection": "images"},
            ),
            Media(
                id="video1",
                type=MediaType.VIDEO,
                url="https://example.com/video.mp4",
                media_bytes=b"video data\x00\xff",
                metadata={"collection": "videos"},
            ),
        ]
        doc = Document(text="Mixed media", id="doc1", media=media, metadata={"collection": "documents"})
        expected_bytes = [item.media_bytes for item in media]
        writer = ZstdWriter(
            output_folder=self.tmp_dir,
            output_filename="${collection}/${rank}/${id}.bin.zst",
            max_file_size=-1,
        )

        written_docs = list(writer.run(iter([doc]), rank=7))
        self.assertEqual([item.path for item in media], ["images/00007/image1.bin.zst", "videos/00007/video1.bin.zst"])
        for item in media:
            self.assertEqual(item.offset, 0)
            self.assertGreater(item.length, 0)
            item.media_bytes = None

        reader = ZstdReader(data_folder=self.tmp_dir, workers=2, preserve_order=True)
        read_docs = list(reader.run(iter(written_docs)))
        self.assertEqual(len(read_docs), 1)
        self.assertEqual([item.media_bytes for item in read_docs[0].media], expected_bytes)

    def test_media_filename_explicit_tags_override_metadata(self) -> None:
        """Keep explicit filename arguments higher priority than media metadata."""
        media = Media(id="image1", type=MediaType.IMAGE, url="", metadata={"collection": "images"})
        with ZstdWriter(output_folder=self.tmp_dir, output_filename="${collection}/${rank}/${id}.bin.zst") as writer:
            self.assertEqual(
                writer._get_output_filename(media, rank=7, collection="override", id="custom"),
                "override/00007/custom.bin.zst",
            )
        self.assertEqual(media.metadata, {"collection": "images"})
        self.assertEqual(media.id, "image1")

    def test_media_filename_reserved_tags(self) -> None:
        """Media metadata must not replace the media ID or worker rank."""
        media = Media(id="image1", type=MediaType.IMAGE, url="", metadata={"id": "other", "rank": "other"})
        with ZstdWriter(output_folder=self.tmp_dir, output_filename="${rank}/${id}.bin.zst") as writer:
            self.assertEqual(writer._get_output_filename(media, rank=7), "00007/image1.bin.zst")

    def test_media_filename_unknown_tag(self) -> None:
        """Continue rejecting filename tags absent from media and explicit arguments."""
        media = Media(id="image1", type=MediaType.IMAGE, url="")
        with ZstdWriter(output_folder=self.tmp_dir, output_filename="${missing}.bin.zst") as writer:
            with self.assertRaises(KeyError):
                writer._get_output_filename(media)

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
