"""Tests for data.py - Document and Media dataclasses"""

import unittest

from datatrove.data import Document, Media, MediaType


class TestMediaDataclass(unittest.TestCase):
    """Test Media dataclass"""

    def test_media_creation(self):
        """Test basic Media creation"""
        media = Media(id="test1", type=MediaType.IMAGE, url="https://example.com/image.jpg")
        self.assertEqual(media.id, "test1")
        self.assertEqual(media.type, MediaType.IMAGE)
        self.assertEqual(media.url, "https://example.com/image.jpg")
        self.assertIsNone(media.alt)
        self.assertIsNone(media.path)
        self.assertIsNone(media.offset)
        self.assertIsNone(media.length)
        self.assertIsNone(media.media_bytes)
        self.assertEqual(media.metadata, {})

    def test_media_with_all_fields(self):
        """Test Media with all fields populated"""
        media = Media(
            id="test1",
            type=MediaType.VIDEO,
            url="https://example.com/video.mp4",
            alt="Test video",
            path="/storage/video.mp4",
            offset=100,
            length=5000,
            media_bytes=b"test data",
            metadata={"width": 1920, "height": 1080, "duration": 120.5},
        )
        self.assertEqual(media.id, "test1")
        self.assertEqual(media.type, MediaType.VIDEO)
        self.assertEqual(media.url, "https://example.com/video.mp4")
        self.assertEqual(media.alt, "Test video")
        self.assertEqual(media.path, "/storage/video.mp4")
        self.assertEqual(media.offset, 100)
        self.assertEqual(media.length, 5000)
        self.assertEqual(media.media_bytes, b"test data")
        self.assertEqual(media.metadata["width"], 1920)
        self.assertEqual(media.metadata["height"], 1080)
        self.assertEqual(media.metadata["duration"], 120.5)

    def test_media_metadata_types(self):
        """Test that Media metadata accepts correct types"""
        media = Media(
            id="test1",
            type=MediaType.IMAGE,
            url="https://example.com/image.jpg",
            metadata={"string_val": "test", "int_val": 42, "float_val": 3.14, "bool_val": True},
        )
        self.assertEqual(media.metadata["string_val"], "test")
        self.assertEqual(media.metadata["int_val"], 42)
        self.assertEqual(media.metadata["float_val"], 3.14)
        self.assertEqual(media.metadata["bool_val"], True)


class TestDocumentDataclass(unittest.TestCase):
    """Test Document dataclass"""

    def test_document_creation(self):
        """Test basic Document creation"""
        doc = Document(text="Test document", id="doc1")
        self.assertEqual(doc.text, "Test document")
        self.assertEqual(doc.id, "doc1")
        self.assertEqual(doc.media, [])
        self.assertEqual(doc.metadata, {})

    def test_document_with_media(self):
        """Test Document with media"""
        media = Media(id="media1", type=MediaType.IMAGE, url="https://example.com/image.jpg")
        doc = Document(text="Test document", id="doc1", media=[media])
        self.assertEqual(len(doc.media), 1)
        self.assertEqual(doc.media[0].id, "media1")
        self.assertEqual(doc.media[0].type, MediaType.IMAGE)

    def test_document_with_multiple_media(self):
        """Test Document with multiple media items"""
        media1 = Media(id="m1", type=MediaType.IMAGE, url="url1")
        media2 = Media(id="m2", type=MediaType.VIDEO, url="url2")
        media3 = Media(id="m3", type=MediaType.AUDIO, url="url3")

        doc = Document(text="Test", id="doc1", media=[media1, media2, media3])
        self.assertEqual(len(doc.media), 3)
        self.assertEqual([m.id for m in doc.media], ["m1", "m2", "m3"])

    def test_document_metadata_types(self):
        """Test that Document metadata accepts correct types"""
        doc = Document(
            text="Test", id="doc1", metadata={"source": "test", "line_number": 42, "score": 0.95, "is_valid": True}
        )
        self.assertEqual(doc.metadata["source"], "test")
        self.assertEqual(doc.metadata["line_number"], 42)
        self.assertEqual(doc.metadata["score"], 0.95)
        self.assertEqual(doc.metadata["is_valid"], True)


class TestMediaType(unittest.TestCase):
    """Test MediaType constants"""

    def test_media_types(self):
        """Test all MediaType constants"""
        self.assertEqual(MediaType.IMAGE, 0)
        self.assertEqual(MediaType.VIDEO, 1)
        self.assertEqual(MediaType.AUDIO, 2)
        self.assertEqual(MediaType.DOCUMENT, 3)

    def test_media_type_uniqueness(self):
        """Test that MediaType values are unique"""
        types = [MediaType.IMAGE, MediaType.VIDEO, MediaType.AUDIO, MediaType.DOCUMENT]
        self.assertEqual(len(types), len(set(types)))


class TestDocumentIntegration(unittest.TestCase):
    """Integration tests for Document and Media"""

    def test_document_with_media_bytes(self):
        """Test Document with Media containing bytes"""
        media = Media(
            id="img1", type=MediaType.IMAGE, url="https://example.com/image.jpg", media_bytes=b"fake image data"
        )
        doc = Document(text="Document with image", id="doc1", media=[media], metadata={"source": "test"})

        self.assertEqual(doc.media[0].media_bytes, b"fake image data")
        self.assertEqual(len(doc.media[0].media_bytes), 15)

    def test_document_media_path_offset_length(self):
        """Test Document with Media path, offset, and length"""
        media = Media(
            id="img1",
            type=MediaType.IMAGE,
            url="https://example.com/image.jpg",
            path="/storage/images.bin.zst",
            offset=1024,
            length=4096,
        )
        doc = Document(text="Document", id="doc1", media=[media])

        self.assertEqual(doc.media[0].path, "/storage/images.bin.zst")
        self.assertEqual(doc.media[0].offset, 1024)
        self.assertEqual(doc.media[0].length, 4096)

    def test_empty_document(self):
        """Test Document with empty text"""
        doc = Document(text="", id="doc1")
        self.assertEqual(doc.text, "")
        self.assertEqual(doc.id, "doc1")

    def test_document_with_long_text(self):
        """Test Document with long text"""
        long_text = "x" * 1000000  # 1MB of text
        doc = Document(text=long_text, id="doc1")
        self.assertEqual(len(doc.text), 1000000)

    def test_document_with_unicode(self):
        """Test Document with Unicode text"""
        doc = Document(text="Hello 世界 🌍 Здравствуй мир", id="doc1", metadata={"language": "multi"})
        self.assertIn("世界", doc.text)
        self.assertIn("🌍", doc.text)
        self.assertIn("Здравствуй", doc.text)


if __name__ == "__main__":
    unittest.main()
