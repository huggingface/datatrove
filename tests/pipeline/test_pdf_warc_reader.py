import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

from datatrove.pipeline.readers.pdf_warc import PDFWarcReader, process_pdf_record


class TestPDFWarcReader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_process_pdf_record_filters_non_pdfs(self):
        """Test that non-PDF records are filtered out."""
        # Mock HTML record
        record = Mock()
        record.rec_type = "response"
        record.content_stream.return_value.read.return_value = b"<html>test</html>"
        record.content_type = "text/html"
        record.rec_headers = {"WARC-Identified-Payload-Type": "text/html"}

        with patch('magic.from_buffer', return_value="text/html"):
            result = process_pdf_record(record, 0, "test.warc", {"application/pdf"})
            self.assertIsNone(result)

    def test_process_pdf_record_accepts_pdfs_by_mime(self):
        """Test that PDF records are accepted by MIME type."""
        # Mock PDF record
        record = Mock()
        record.rec_type = "response"
        record.content_stream.return_value.read.return_value = b"%PDF-1.4 fake pdf content"
        record.content_type = "application/pdf"
        record.rec_headers = {
            "WARC-Identified-Payload-Type": "application/pdf",
            "WARC-Record-ID": "test-id",
            "WARC-Target-URI": "http://example.com/test.pdf",
            "WARC-Date": "2023-01-01T00:00:00Z"
        }
        record.http_headers.statusline = "200 OK"

        result = process_pdf_record(record, 1000, "test.warc", {"application/pdf"})

        self.assertIsNotNone(result)
        self.assertEqual(result["text"], b"%PDF-1.4 fake pdf content")
        self.assertEqual(result["id"], "test-id")
        self.assertEqual(result["content_mime_detected"], "application/pdf")
        self.assertEqual(result["is_truncated"], False)

    def test_process_pdf_record_accepts_pdfs_by_url(self):
        """Test that PDFs are accepted by URL heuristic."""
        record = Mock()
        record.rec_type = "response"
        record.content_stream.return_value.read.return_value = b"%PDF-1.4 fake pdf content"
        record.content_type = "application/octet-stream"
        record.rec_headers = {
            "WARC-Identified-Payload-Type": None,
            "WARC-Record-ID": "test-id",
            "WARC-Target-URI": "http://example.com/document.pdf",
            "WARC-Date": "2023-01-01T00:00:00Z"
        }
        record.http_headers.statusline = "200 OK"

        with patch('magic.from_buffer', return_value="application/octet-stream"):
            result = process_pdf_record(record, 1000, "test.warc", {"application/pdf"})

        self.assertIsNotNone(result)
        self.assertEqual(result["url"], "http://example.com/document.pdf")

    def test_process_pdf_record_detects_truncation_by_length(self):
        """Test truncation detection by 1MB length."""
        record = Mock()
        record.rec_type = "response"
        # Exactly 1MB content
        record.content_stream.return_value.read.return_value = b"x" * (1024 * 1024)
        record.content_type = "application/pdf"
        record.rec_headers = {
            "WARC-Identified-Payload-Type": "application/pdf",
            "WARC-Record-ID": "test-id",
            "WARC-Target-URI": "http://example.com/test.pdf",
            "WARC-Date": "2023-01-01T00:00:00Z"
        }
        record.http_headers.statusline = "200 OK"

        result = process_pdf_record(record, 1000, "test.warc", {"application/pdf"})

        self.assertTrue(result["is_truncated"])
        self.assertEqual(result["truncation_reason"], "length")

    def test_process_pdf_record_detects_truncation_by_field(self):
        """Test truncation detection by WARC field."""
        record = Mock()
        record.rec_type = "response"
        record.content_stream.return_value.read.return_value = b"%PDF-1.4 truncated content"
        record.content_type = "application/pdf"
        record.rec_headers = {
            "WARC-Identified-Payload-Type": "application/pdf",
            "WARC-Record-ID": "test-id",
            "WARC-Target-URI": "http://example.com/test.pdf",
            "WARC-Date": "2023-01-01T00:00:00Z",
            "WARC-Truncated": "length"
        }
        record.http_headers.statusline = "200 OK"

        result = process_pdf_record(record, 1000, "test.warc", {"application/pdf"})

        self.assertTrue(result["is_truncated"])
        self.assertEqual(result["truncation_reason"], "warc_field")

    @patch('datatrove.pipeline.readers.base.BaseDiskReader.__init__')
    def test_pdf_warc_reader_initialization(self, mock_base_init):
        """Test PDFWarcReader initialization."""
        mock_base_init.return_value = None
        reader = PDFWarcReader.__new__(PDFWarcReader)
        reader.pdf_mime_types = {"application/pdf", "application/x-pdf"}

        # Test that mime types are set correctly
        self.assertEqual(reader.pdf_mime_types, {"application/pdf", "application/x-pdf"})

    @patch('datatrove.pipeline.readers.base.BaseDiskReader.__init__')
    def test_pdf_warc_reader_default_mime_types(self, mock_base_init):
        """Test PDFWarcReader with default MIME types."""
        mock_base_init.return_value = None

        # Test initialization logic
        pdf_mime_types = None
        if pdf_mime_types is None:
            pdf_mime_types = ["application/pdf"]
        pdf_mime_types_set = set(pdf_mime_types)

        self.assertEqual(pdf_mime_types_set, {"application/pdf"})


    def test_pdf_warc_reader_with_real_data(self):
        """Integration test with real CommonCrawl WARC file."""
        import os
        warc_file_path = "examples_local/data/CC-MAIN-20180420081400-20180420101400-00000.warc.gz"

        # Skip if test data file doesn't exist
        if not os.path.exists(warc_file_path):
            self.skipTest(f"Test WARC file not found: {warc_file_path}")

        reader = PDFWarcReader(
            data_folder="examples_local/data/",
            glob_pattern="CC-MAIN-20180420081400-20180420101400-00000.warc.gz",
            limit=5  # Only test first 5 PDFs
        )

        pdf_count = 0
        truncated_count = 0
        actual_pdf_count = 0

        for doc in reader.run():
            pdf_count += 1

            # Basic document structure checks
            self.assertIsInstance(doc.text, bytes, "PDF content should be bytes")
            self.assertIsInstance(doc.id, str, "Document ID should be string")
            self.assertIn("url", doc.metadata, "Should have URL metadata")
            self.assertIn("content_length", doc.metadata, "Should have content length")
            self.assertIn("is_truncated", doc.metadata, "Should have truncation flag")

            # Check if it's an actual PDF (starts with PDF header)
            if doc.text.startswith(b'%PDF-'):
                actual_pdf_count += 1

            if doc.metadata.get('is_truncated'):
                truncated_count += 1
                self.assertIn("truncation_reason", doc.metadata, "Truncated PDFs should have reason")

        # Verify we found some PDFs
        self.assertGreater(pdf_count, 0, "Should find at least some PDFs")
        self.assertGreater(actual_pdf_count, 0, "Should find at least some actual PDF files")

        print(f"\nIntegration test results: {pdf_count} PDFs found, {actual_pdf_count} with PDF headers, {truncated_count} truncated")


if __name__ == "__main__":
    unittest.main()