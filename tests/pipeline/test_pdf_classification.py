import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from datatrove.data import Document
from datatrove.pipeline.media.predictor.scanned_pdf_predictor import (
    PDFScannedPredictor,
    PDFFeatureExtractor,
    flatten_per_page_features
)


class TestPDFClassification(unittest.TestCase):

    def setUp(self):
        # Create a simple XGBoost model for testing
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix='.xgb', delete=False)
        self.temp_model_file.close()

        # Train a simple model with dummy data
        self._create_test_model()

        # Sample PDF content
        self.valid_pdf_content = (
            b'%PDF-1.4\n'
            b'1 0 obj\n'
            b'<<\n/Type /Catalog\n/Pages 2 0 R\n>>\n'
            b'endobj\n'
            b'2 0 obj\n'
            b'<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\n'
            b'endobj\n'
            b'3 0 obj\n'
            b'<<\n/Type /Page\n/Parent 2 0 R\n/Resources <<\n/Font <<\n/F1 4 0 R\n>>\n>>\n'
            b'/MediaBox [0 0 612 792]\n/Contents 5 0 R\n>>\n'
            b'endobj\n'
            b'4 0 obj\n'
            b'<<\n/Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n>>\n'
            b'endobj\n'
            b'5 0 obj\n'
            b'<<\n/Length 44\n>>\n'
            b'stream\n'
            b'BT\n/F1 12 Tf\n100 700 Td\n(Hello World) Tj\nET\n'
            b'endstream\n'
            b'endobj\n'
            b'xref\n'
            b'0 6\n'
            b'0000000000 65535 f \n'
            b'0000000010 00000 n \n'
            b'0000000079 00000 n \n'
            b'0000000173 00000 n \n'
            b'0000000301 00000 n \n'
            b'0000000380 00000 n \n'
            b'trailer\n'
            b'<<\n/Size 6\n/Root 1 0 R\n>>\n'
            b'startxref\n'
            b'492\n'
            b'%%EOF\n'
        )

    def tearDown(self):
        if os.path.exists(self.temp_model_file.name):
            os.unlink(self.temp_model_file.name)

    def _create_test_model(self):
        """Create a simple trained XGBoost model for testing."""
        # Create dummy training data with 124 features (real data dimension)
        np.random.seed(42)
        n_samples = 100
        n_features = 124  # Based on real CommonCrawl data analysis

        # Generate dummy features
        X = np.random.random((n_samples, n_features))

        # Generate labels (0 = text-based, 1 = OCR needed)
        # Make it deterministic based on some features
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

        # Train model
        model = XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X, y)

        # Save model
        model.save_model(self.temp_model_file.name)

    def test_feature_extractor_initialization(self):
        """Test PDFFeatureExtractor initialization."""
        extractor = PDFFeatureExtractor(num_pages_to_sample=5)
        self.assertEqual(extractor.num_pages_to_sample, 5)
        self.assertEqual(extractor.num_chunks, 1)

    def test_flatten_per_page_features(self):
        """Test feature flattening logic."""
        # Create mock feature dictionary
        feature_dict = {
            # Document level features
            "num_pages_successfully_sampled": 3,
            "num_unique_image_xrefs": 5,
            "num_junk_image_xrefs": 1,
            "garbled_text_ratio": 0.1,
            "is_form": False,
            "creator_or_producer_is_known_scanner": True,
            "class": 1,

            # Page level features (3 pages of data)
            "page_level_unique_font_counts": [2, 3, 1],
            "page_level_char_counts": [100, 150, 80],
            "page_level_text_box_counts": [5, 7, 3],
            "page_level_avg_text_box_lengths": [20.0, 25.0, 15.0],
            "page_level_text_area_ratios": [0.3, 0.4, 0.2],
            "page_level_hidden_char_counts": [0, 0, 0],
            "page_level_hidden_text_box_counts": [0, 0, 0],
            "page_level_hidden_avg_text_box_lengths": [0.0, 0.0, 0.0],
            "page_level_hidden_text_area_ratios": [0.0, 0.0, 0.0],
            "page_level_image_counts": [2, 1, 3],
            "page_level_non_junk_image_counts": [2, 1, 2],
            "page_level_bitmap_proportions": [0.1, 0.05, 0.15],
            "page_level_max_merged_strip_areas": [0.2, 0.1, 0.25],
            "page_level_drawing_strokes_count": [5, 3, 8],
            "page_level_vector_graphics_obj_count": [2, 1, 4],
        }

        flattened = flatten_per_page_features(feature_dict, sample_to_k_page_features=8)

        # Check document level features are preserved
        self.assertEqual(flattened["num_pages_successfully_sampled"], 3)
        self.assertEqual(flattened["is_form"], False)
        self.assertTrue(flattened["creator_or_producer_is_known_scanner"])

        # Check page level features are flattened (should have 8 pages worth)
        self.assertIn("page_level_unique_font_counts_page1", flattened)
        self.assertIn("page_level_unique_font_counts_page8", flattened)

        # Should have exactly the right number of features
        expected_doc_features = 7  # From the list in the function
        expected_page_features = 15 * 8  # 15 page features × 8 pages
        expected_total = expected_doc_features + expected_page_features
        self.assertEqual(len(flattened), expected_total)

    @patch('pymupdf.open')
    def test_pdf_scanned_predictor_initialization(self, mock_pymupdf_open):
        """Test PDFScannedPredictor initialization."""
        predictor = PDFScannedPredictor(
            path_to_model=self.temp_model_file.name,
            num_pages_to_sample=8
        )

        self.assertEqual(predictor.path_to_model, self.temp_model_file.name)
        self.assertEqual(predictor.num_pages_to_sample, 8)
        self.assertIsNone(predictor._model)  # Lazy loading

    @patch('pymupdf.open')
    def test_pdf_scanned_predictor_model_loading(self, mock_pymupdf_open):
        """Test XGBoost model loading."""
        predictor = PDFScannedPredictor(
            path_to_model=self.temp_model_file.name,
            num_pages_to_sample=8
        )

        # Access model property to trigger loading
        model = predictor.model
        self.assertIsNotNone(model)
        self.assertIsInstance(model, XGBClassifier)

    def test_pdf_scanned_predictor_with_real_pdf(self):
        """Test prediction with a mock feature extraction."""
        predictor = PDFScannedPredictor(
            path_to_model=self.temp_model_file.name,
            num_pages_to_sample=8
        )

        # Mock the extractor run method to return valid features
        mock_features = {
            # Document level features
            "num_pages_successfully_sampled": 1,
            "num_unique_image_xrefs": 0,
            "num_junk_image_xrefs": 0,
            "garbled_text_ratio": 0.0,
            "is_form": False,
            "creator_or_producer_is_known_scanner": False,
            "class": 0,
            # Page level features (1 page of dummy data)
            "page_level_unique_font_counts": [2],
            "page_level_char_counts": [100],
            "page_level_text_box_counts": [5],
            "page_level_avg_text_box_lengths": [20.0],
            "page_level_text_area_ratios": [0.3],
            "page_level_hidden_char_counts": [0],
            "page_level_hidden_text_box_counts": [0],
            "page_level_hidden_avg_text_box_lengths": [0.0],
            "page_level_hidden_text_area_ratios": [0.0],
            "page_level_image_counts": [0],
            "page_level_non_junk_image_counts": [0],
            "page_level_bitmap_proportions": [0.0],
            "page_level_max_merged_strip_areas": [0.0],
            "page_level_drawing_strokes_count": [0],
            "page_level_vector_graphics_obj_count": [0],
        }

        with patch('pymupdf.open') as mock_pymupdf_open, \
             patch.object(predictor.extractor, 'run') as mock_extractor_run:

            # Mock PyMuPDF document
            mock_doc = mock_pymupdf_open.return_value.__enter__.return_value
            mock_doc.__len__.return_value = 1
            mock_doc.is_encrypted = False
            mock_doc.needs_pass = False

            # Mock feature extraction to return our test features
            mock_extractor_run.return_value = [mock_features]

            result = predictor.predict(self.valid_pdf_content)

        # Check result structure
        self.assertIn("ocr_prob", result)
        self.assertIn("is_form", result)
        self.assertIn("garbled_text_ratio", result)
        self.assertIn("is_encrypted", result)
        self.assertIn("needs_password", result)
        self.assertIn("num_pages", result)

        # Check data types
        self.assertIsInstance(result["ocr_prob"], float)
        self.assertIsInstance(result["is_form"], bool)
        self.assertIsInstance(result["garbled_text_ratio"], float)
        self.assertIsInstance(result["is_encrypted"], bool)
        self.assertIsInstance(result["needs_password"], bool)
        self.assertIsInstance(result["num_pages"], int)

        # Check reasonable ranges
        self.assertGreaterEqual(result["ocr_prob"], 0.0)
        self.assertLessEqual(result["ocr_prob"], 1.0)
        self.assertGreaterEqual(result["garbled_text_ratio"], 0.0)
        self.assertLessEqual(result["garbled_text_ratio"], 1.0)

    def test_pdf_scanned_predictor_handles_none_bytes(self):
        """Test predictor handles None input gracefully."""
        predictor = PDFScannedPredictor(
            path_to_model=self.temp_model_file.name,
            num_pages_to_sample=8
        )

        result = predictor.predict(None)
        self.assertIn("prediction_failed", result)
        self.assertEqual(result["prediction_failed"], "Media bytes are None")

    @patch('pymupdf.open')
    def test_pdf_scanned_predictor_handles_invalid_pdf(self, mock_pymupdf_open):
        """Test predictor handles invalid PDF gracefully."""
        mock_pymupdf_open.side_effect = Exception("Invalid PDF")

        predictor = PDFScannedPredictor(
            path_to_model=self.temp_model_file.name,
            num_pages_to_sample=8
        )

        result = predictor.predict(b"not a pdf")
        self.assertIn("prediction_failed", result)
        self.assertEqual(result["prediction_failed"], "Invalid PDF")

    def test_pdf_feature_extraction_with_real_data(self):
        """Integration test with real CommonCrawl PDF data."""
        import sys
        sys.path.insert(0, 'src')
        from datatrove.pipeline.readers.pdf_warc import PDFWarcReader

        warc_file_path = "examples_local/data/CC-MAIN-20180420081400-20180420101400-00000.warc.gz"

        # Skip if test data file doesn't exist
        if not os.path.exists(warc_file_path):
            self.skipTest(f"Test WARC file not found: {warc_file_path}")

        reader = PDFWarcReader(
            data_folder="examples_local/data/",
            glob_pattern="CC-MAIN-20180420081400-20180420101400-00000.warc.gz",
            limit=3  # Test on small subset
        )

        predictor = PDFScannedPredictor(
            path_to_model=self.temp_model_file.name,
            num_pages_to_sample=5  # Smaller for faster testing
        )

        pdf_count = 0
        successful_predictions = 0
        failed_predictions = 0

        for doc in reader.run():
            pdf_count += 1

            # Only test PDFs with valid headers
            if not doc.text.startswith(b'%PDF-'):
                continue

            result = predictor.predict(doc.text)

            if "prediction_failed" in result:
                failed_predictions += 1
                print(f"Prediction failed for PDF {doc.id}: {result['prediction_failed']}")
            else:
                successful_predictions += 1
                print(f"PDF {doc.id}: OCR prob={result['ocr_prob']:.3f}, "
                      f"form={result['is_form']}, "
                      f"garbled={result['garbled_text_ratio']:.3f}, "
                      f"pages={result['num_pages']}")

        # Verify we processed some PDFs
        self.assertGreater(pdf_count, 0, "Should find at least some PDFs")

        # We expect some predictions to work, but failures are OK with real data
        total_processed = successful_predictions + failed_predictions
        if total_processed > 0:
            success_rate = successful_predictions / total_processed
            print(f"\nClassification results: {successful_predictions}/{total_processed} successful "
                  f"({success_rate:.1%} success rate)")

    def test_real_trained_model(self):
        """Test with the actual trained model from CommonCrawl data."""
        import sys
        sys.path.insert(0, 'examples_local')

        real_model_path = "examples_local/pdf_classifier_real_data.xgb"

        # Skip if real model doesn't exist
        if not os.path.exists(real_model_path):
            self.skipTest(f"Real trained model not found: {real_model_path}")

        predictor = PDFScannedPredictor(
            path_to_model=real_model_path,
            num_pages_to_sample=8
        )

        # Test with real WARC data
        sys.path.insert(0, 'src')
        from datatrove.pipeline.readers.pdf_warc import PDFWarcReader

        warc_file_path = "examples_local/data/CC-MAIN-20180420081400-20180420101400-00000.warc.gz"
        if not os.path.exists(warc_file_path):
            self.skipTest(f"Test WARC file not found: {warc_file_path}")

        reader = PDFWarcReader(
            data_folder="examples_local/data/",
            glob_pattern="CC-MAIN-20180420081400-20180420101400-00000.warc.gz",
            limit=10  # Test on small subset
        )

        predictions = []
        processed = 0

        for doc in reader.run():
            if not doc.text.startswith(b'%PDF-'):
                continue

            result = predictor.predict(doc.text)

            if "prediction_failed" not in result:
                predictions.append({
                    'id': doc.id,
                    'ocr_prob': result['ocr_prob'],
                    'is_form': result['is_form'],
                    'garbled_ratio': result['garbled_text_ratio'],
                    'num_pages': result['num_pages']
                })
                processed += 1

                if processed >= 5:  # Limit for test
                    break

        # Verify predictions
        self.assertGreater(len(predictions), 0, "Should get some predictions")

        for pred in predictions:
            # Check prediction structure
            self.assertIn('ocr_prob', pred)
            self.assertIsInstance(pred['ocr_prob'], float)
            self.assertGreaterEqual(pred['ocr_prob'], 0.0)
            self.assertLessEqual(pred['ocr_prob'], 1.0)

        print(f"\nReal model test results ({len(predictions)} PDFs):")
        for i, pred in enumerate(predictions):
            decision = "OCR" if pred['ocr_prob'] > 0.5 else "Text"
            print(f"  PDF {i+1}: {decision} ({pred['ocr_prob']:.3f}), "
                  f"form={pred['is_form']}, pages={pred['num_pages']}")

        avg_ocr_prob = sum(p['ocr_prob'] for p in predictions) / len(predictions)
        print(f"  Average OCR probability: {avg_ocr_prob:.3f}")


if __name__ == "__main__":
    unittest.main()