import unittest

from datatrove.data import Document
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.pipeline.formatters.ftfy import FTFYFormatter


class UpperCaseFormatter(BaseFormatter):
    """Concrete formatter for testing the BaseFormatter run() pipeline."""

    name = "🔠 UpperCase"

    def format(self, text: str) -> str:
        return text.upper()


class TestBaseFormatter(unittest.TestCase):
    def test_run_applies_format_and_tracks_stats(self):
        """Verify run() mutates text in-place and counts stats correctly."""
        formatter = UpperCaseFormatter()
        docs = [
            Document(text="hello world", id="1", metadata={"lang": "en"}),
            Document(text="foo bar", id="2"),
        ]
        result = list(formatter.run(iter(docs)))
        assert result[0].text == "HELLO WORLD"
        assert result[1].text == "FOO BAR"
        # Metadata must survive the pipeline
        assert result[0].metadata["lang"] == "en"
        assert formatter.stats["total"].total == 2


class TestFTFYFormatter(unittest.TestCase):
    def test_fixes_mojibake_encoding(self):
        formatter = FTFYFormatter()
        broken = "l\u00e2\u0080\u0099intelligence"
        fixed = formatter.format(broken)
        assert "\u00e2" not in fixed

    def test_removes_control_chars(self):
        formatter = FTFYFormatter()
        assert "\x00" not in formatter.format("hello\x00world")

    def test_preserves_normal_text(self):
        formatter = FTFYFormatter()
        text = "Hello, this is a normal sentence."
        assert formatter.format(text) == text

    def test_unescape_html_entities(self):
        formatter = FTFYFormatter(unescape_html=True)
        fixed = formatter.format("price &gt; 10")
        assert ">" in fixed

    def test_run_pipeline_end_to_end(self):
        formatter = FTFYFormatter()
        doc = Document(text="hello\x00world", id="1")
        result = list(formatter.run(iter([doc])))
        assert "\x00" not in result[0].text


if __name__ == "__main__":
    unittest.main()
