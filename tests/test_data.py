import unittest

from datatrove.data import Document, Media


class TestMedia(unittest.TestCase):
    def test_metadata_default_factory_is_independent(self):
        """Mutable default fields must not be shared across instances."""
        m1 = Media(id="1", type=0, url="a")
        m2 = Media(id="2", type=0, url="b")
        m1.metadata["key"] = "value"
        assert "key" not in m2.metadata


class TestDocument(unittest.TestCase):
    def test_metadata_default_factory_is_independent(self):
        """Mutable default fields must not be shared across instances."""
        d1 = Document(text="a", id="1")
        d2 = Document(text="b", id="2")
        d1.metadata["key"] = "value"
        assert "key" not in d2.metadata

    def test_media_default_factory_is_independent(self):
        """Mutable default fields must not be shared across instances."""
        d1 = Document(text="a", id="1")
        d2 = Document(text="b", id="2")
        d1.media.append(Media(id="img1", type=0, url="x"))
        assert len(d2.media) == 0


if __name__ == "__main__":
    unittest.main()
