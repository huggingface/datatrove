"""Tests for media utility functions"""
import unittest

from datatrove.data import Document, Media, MediaType
from datatrove.utils.media import iter_pages


class TestMediaUtils(unittest.TestCase):
    """Test media utility functions"""

    def test_iter_pages_single_media(self):
        """Test iter_pages with a single media item"""
        text = "Page 1 content. Page 2 content. Page 3 content."
        doc = Document(
            text=text,
            id="doc1",
            media=[
                Media(
                    id="media1",
                    type=MediaType.DOCUMENT,
                    url="https://example.com/doc.pdf",
                    metadata={"page_offsets": [15, 31, 47]}
                )
            ]
        )

        pages = list(iter_pages(doc))
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[0], "Page 1 content.")
        self.assertEqual(pages[1], " Page 2 content.")
        self.assertEqual(pages[2], " Page 3 content.")

    def test_iter_pages_multiple_media(self):
        """Test iter_pages with multiple media items"""
        text = "Section A text. Section B text. Section C text."
        doc = Document(
            text=text,
            id="doc1",
            media=[
                Media(
                    id="media1",
                    type=MediaType.DOCUMENT,
                    url="https://example.com/doc1.pdf",
                    metadata={"page_offsets": [16]}
                ),
                Media(
                    id="media2",
                    type=MediaType.DOCUMENT,
                    url="https://example.com/doc2.pdf",
                    metadata={"page_offsets": [32, 48]}
                )
            ]
        )

        pages = list(iter_pages(doc))
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[0], "Section A text. ")
        self.assertEqual(pages[1], "Section B text. ")
        self.assertEqual(pages[2], "Section C text.")


    def test_iter_pages_empty_media(self):
        """Test iter_pages with no media"""
        text = "Some text content."
        doc = Document(
            text=text,
            id="doc1",
            media=[]
        )

        pages = list(iter_pages(doc))
        self.assertEqual(len(pages), 0)
