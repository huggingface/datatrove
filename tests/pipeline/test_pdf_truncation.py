import unittest
from datatrove.data import Document
from datatrove.pipeline.filters.pdf_truncation import PDFTruncationFilter, PDFValidationFilter


class TestPDFTruncationFilter(unittest.TestCase):
    def setUp(self):
        self.valid_pdf_content = b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\ntrailer\nxref\n1 0 obj\nendobj'
        self.invalid_pdf_content = b'<html>Not a PDF</html>'
        self.truncated_pdf_content = b'%PDF-1.4 truncated content'

    def test_exclude_action_filters_truncated(self):
        """Test that exclude action removes truncated PDFs."""
        filter_step = PDFTruncationFilter(action="exclude")

        # Truncated document
        doc = Document(
            text=self.valid_pdf_content,
            id="test-1",
            metadata={"is_truncated": True, "truncation_reason": "warc_field"}
        )

        result = filter_step.filter(doc)
        self.assertFalse(result, "Truncated PDF should be excluded")

    def test_exclude_action_keeps_valid(self):
        """Test that exclude action keeps valid PDFs."""
        filter_step = PDFTruncationFilter(action="exclude")

        # Valid document
        doc = Document(
            text=self.valid_pdf_content,
            id="test-1",
            metadata={"is_truncated": False, "content_length": 5000}
        )

        result = filter_step.filter(doc)
        self.assertTrue(result, "Valid PDF should be kept")

    def test_mark_for_refetch_action(self):
        """Test that mark_for_refetch adds refetch metadata."""
        filter_step = PDFTruncationFilter(action="mark_for_refetch")

        # Truncated document
        doc = Document(
            text=self.truncated_pdf_content,
            id="test-1",
            metadata={"is_truncated": True, "content_length": 100}
        )

        result = filter_step.filter(doc)
        self.assertTrue(result, "Truncated PDF should be kept for refetch")
        self.assertTrue(doc.metadata.get("needs_refetch"), "Should be marked for refetch")
        self.assertEqual(doc.metadata.get("original_content_length"), 100)

    def test_include_action_keeps_all(self):
        """Test that include action keeps all PDFs."""
        filter_step = PDFTruncationFilter(action="include")

        # Truncated document
        doc = Document(
            text=self.truncated_pdf_content,
            id="test-1",
            metadata={"is_truncated": True}
        )

        result = filter_step.filter(doc)
        self.assertTrue(result, "All PDFs should be kept with include action")

    def test_check_pdf_header_detection(self):
        """Test PDF header validation."""
        filter_step = PDFTruncationFilter(action="exclude", check_pdf_header=True)

        # Invalid PDF header
        doc = Document(
            text=self.invalid_pdf_content,
            id="test-1",
            metadata={"is_truncated": False}
        )

        result = filter_step.filter(doc)
        self.assertFalse(result, "Invalid PDF header should be excluded")
        self.assertTrue(doc.metadata.get("is_truncated"), "Should be marked as truncated")
        self.assertIn("invalid_pdf_header", doc.metadata.get("truncation_reason", ""))

    def test_min_size_detection(self):
        """Test minimum size validation."""
        filter_step = PDFTruncationFilter(action="exclude", min_size_bytes=100)

        # Too small PDF
        doc = Document(
            text=b'%PDF-1.4\nsmall',
            id="test-1",
            metadata={"is_truncated": False, "content_length": 50}
        )

        result = filter_step.filter(doc)
        self.assertFalse(result, "Too small PDF should be excluded")
        self.assertTrue(doc.metadata.get("is_truncated"), "Should be marked as truncated")
        self.assertIn("too_small", doc.metadata.get("truncation_reason", ""))

    def test_multiple_truncation_reasons(self):
        """Test handling of multiple truncation reasons."""
        filter_step = PDFTruncationFilter(action="mark_for_refetch", check_pdf_header=True, min_size_bytes=100)

        # PDF with multiple issues
        doc = Document(
            text=b'invalid small content',
            id="test-1",
            metadata={"is_truncated": True, "truncation_reason": "warc_field", "content_length": 20}
        )

        result = filter_step.filter(doc)
        self.assertTrue(result, "Should be kept for refetch")

        reasons = doc.metadata.get("truncation_reason", "").split(";")
        self.assertIn("warc_field", reasons)
        self.assertIn("invalid_pdf_header", reasons)
        self.assertIn("too_small", reasons)

    def test_invalid_action_raises_error(self):
        """Test that invalid action parameter raises ValueError."""
        with self.assertRaises(ValueError):
            PDFTruncationFilter(action="invalid_action")


class TestPDFValidationFilter(unittest.TestCase):
    def setUp(self):
        self.valid_pdf_content = b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\ntrailer\nxref\n1 0 obj\nendobj'
        self.invalid_pdf_content = b'<html>Not a PDF</html>'
        self.corrupted_pdf_content = b'%PDF-1.4\n' + b'\x00' * 100 + b'\xff\xff' * 50  # High corruption

    def test_valid_pdf_passes(self):
        """Test that valid PDF passes validation."""
        filter_step = PDFValidationFilter()

        doc = Document(
            text=self.valid_pdf_content,
            id="test-1"
        )

        result = filter_step.filter(doc)
        self.assertTrue(result, "Valid PDF should pass validation")
        self.assertTrue(doc.metadata.get("pdf_validation_passed"))

    def test_invalid_header_fails(self):
        """Test that invalid PDF header fails validation."""
        filter_step = PDFValidationFilter()

        doc = Document(
            text=self.invalid_pdf_content,
            id="test-1"
        )

        result = filter_step.filter(doc)
        self.assertFalse(result, "Invalid PDF header should fail validation")

    def test_missing_structure_elements(self):
        """Test detection of missing PDF structure elements."""
        filter_step = PDFValidationFilter(check_pdf_structure=True)

        # PDF without trailer
        doc = Document(
            text=b'%PDF-1.4\nxref\n1 0 obj\nendobj',
            id="test-1"
        )

        result = filter_step.filter(doc)
        self.assertFalse(result, "PDF without trailer should fail validation")
        self.assertTrue(doc.metadata.get("pdf_validation_failed"))
        self.assertIn("missing_trailer", doc.metadata.get("pdf_validation_issues", ""))

    def test_high_corruption_ratio(self):
        """Test detection of highly corrupted PDFs."""
        filter_step = PDFValidationFilter(max_corruption_ratio=0.05)

        doc = Document(
            text=self.corrupted_pdf_content,
            id="test-1"
        )

        result = filter_step.filter(doc)
        self.assertFalse(result, "Highly corrupted PDF should fail validation")
        self.assertTrue(doc.metadata.get("pdf_validation_failed"))
        validation_issues = doc.metadata.get("pdf_validation_issues", "")
        self.assertTrue(any("high_corruption_ratio" in issue for issue in validation_issues.split(";")))

    def test_non_bytes_content_fails(self):
        """Test that non-bytes content fails validation."""
        filter_step = PDFValidationFilter()

        doc = Document(
            text="String content instead of bytes",
            id="test-1"
        )

        result = filter_step.filter(doc)
        self.assertFalse(result, "Non-bytes content should fail validation")


    def test_truncation_filter_with_real_data(self):
        """Integration test with real CommonCrawl data."""
        import os
        import sys
        sys.path.insert(0, 'src')
        from datatrove.pipeline.readers.pdf_warc import PDFWarcReader

        warc_file_path = "examples_local/data/CC-MAIN-20180420081400-20180420101400-00000.warc.gz"

        # Skip if test data file doesn't exist
        if not os.path.exists(warc_file_path):
            self.skipTest(f"Test WARC file not found: {warc_file_path}")

        # Test exclude filter behavior
        reader = PDFWarcReader(
            data_folder="examples_local/data/",
            glob_pattern="CC-MAIN-20180420081400-20180420101400-00000.warc.gz",
            limit=20  # Test on subset
        )

        exclude_filter = PDFTruncationFilter(action="exclude")
        kept_count = 0
        excluded_count = 0

        for doc in reader.run():
            result = exclude_filter.filter(doc)
            if result:
                kept_count += 1
            else:
                excluded_count += 1

        # Test mark for refetch behavior
        reader2 = PDFWarcReader(
            data_folder="examples_local/data/",
            glob_pattern="CC-MAIN-20180420081400-20180420101400-00000.warc.gz",
            limit=20
        )

        refetch_filter = PDFTruncationFilter(action="mark_for_refetch")
        refetch_count = 0
        total_processed = 0

        for doc in reader2.run():
            result = refetch_filter.filter(doc)
            total_processed += 1
            if doc.metadata.get("needs_refetch"):
                refetch_count += 1

        # Assertions
        self.assertGreater(kept_count + excluded_count, 0, "Should process some PDFs")
        self.assertGreater(total_processed, 0, "Should process PDFs for refetch test")
        self.assertEqual(total_processed, kept_count + excluded_count, "Total counts should match")

        print(f"\nIntegration test results:")
        print(f"  Exclude filter: {kept_count} kept, {excluded_count} excluded")
        print(f"  Refetch filter: {refetch_count} marked for refetch out of {total_processed}")


if __name__ == "__main__":
    unittest.main()