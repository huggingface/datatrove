#!/usr/bin/env python3
"""
Example 08b: PDF Feature Analysis

Analyzes PDF features from CommonCrawl WARC files for model training insights.

Components:
- PDFWarcReader: Read PDFs from WARC files
- PDFFeatureExtractor: Extract PDF features
- PDFScannedPredictor: Test classifier predictions

Usage:
    python spec/phase3/examples/08b_pdf_feature_analysis.py --limit 100
"""

import io
import os
import tempfile
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from datatrove.pipeline.media.predictor.scanned_pdf_predictor import (
    PDFFeatureExtractor,
    PDFScannedPredictor,
    flatten_per_page_features,
)
from datatrove.pipeline.readers.pdf_warc import PDFWarcReader
from datatrove.utils.logging import logger


class PDFFeatureAnalyzer:
    def __init__(self, data_folder="spec/phase3/data", limit_per_file=None):
        self.data_folder = data_folder
        self.limit_per_file = limit_per_file
        self.results = {
            'pdf_count': 0,
            'valid_pdf_count': 0,
            'feature_extraction_success': 0,
            'feature_extraction_failures': 0,
            'page_counts': [],
            'feature_dimensions': [],
            'doc_features': [],
            'truncation_stats': defaultdict(int),
            'processing_times': [],
            'errors': defaultdict(int)
        }

        # Create temporary model for testing
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix='.xgb', delete=False)
        self.temp_model_file.close()
        self._create_test_model()

    def _create_test_model(self):
        """Create a simple trained XGBoost model for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 79  # Based on observed real data

        X = np.random.random((n_samples, n_features))
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

        model = XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X, y)
        model.save_model(self.temp_model_file.name)

    def analyze_warc_files(self):
        """Process all WARC files and collect statistics."""
        warc_files = [f for f in os.listdir(self.data_folder) if f.endswith('.warc.gz')]
        logger.info(f"Found {len(warc_files)} WARC files:")
        for f in warc_files:
            logger.info(f"  - {f}")

        for warc_file in warc_files:
            logger.info(f"\nProcessing {warc_file}...")
            self._analyze_single_warc(warc_file)

        self._generate_report()

    def _analyze_single_warc(self, warc_file):
        """Analyze a single WARC file."""
        reader = PDFWarcReader(
            data_folder=self.data_folder,
            glob_pattern=warc_file,
            limit=self.limit_per_file or -1
        )

        # Create feature extractor
        feature_extractor = PDFFeatureExtractor(num_pages_to_sample=8)

        file_pdf_count = 0
        file_start_time = time.time()

        for doc in reader.run():
            self.results['pdf_count'] += 1
            file_pdf_count += 1

            # Check if it's a valid PDF
            if not doc.text.startswith(b'%PDF-'):
                self.results['errors']['invalid_pdf_header'] += 1
                continue

            self.results['valid_pdf_count'] += 1

            # Track truncation status
            if doc.metadata.get('is_truncated'):
                reason = doc.metadata.get('truncation_reason', 'unknown')
                self.results['truncation_stats'][reason] += 1
            else:
                self.results['truncation_stats']['not_truncated'] += 1

            # Try feature extraction
            start_time = time.time()
            try:
                import pymupdf
                pymupdf_doc = pymupdf.open(stream=io.BytesIO(doc.text), filetype="pdf")

                # Extract features
                features_raw = feature_extractor.run(pymupdf_doc)

                if features_raw:
                    features = features_raw[0]  # First chunk

                    # Record statistics
                    num_pages = features.get('num_pages_successfully_sampled', 0)
                    self.results['page_counts'].append(num_pages)

                    # Flatten features to see final dimension
                    try:
                        flattened = flatten_per_page_features(features, sample_to_k_page_features=8)
                        self.results['feature_dimensions'].append(len(flattened))

                        # Store some document-level features for analysis
                        doc_feature_summary = {
                            'num_pages': num_pages,
                            'is_form': features.get('is_form', False),
                            'garbled_text_ratio': features.get('garbled_text_ratio', 0.0),
                            'creator_or_producer_is_known_scanner': features.get('creator_or_producer_is_known_scanner', False),
                            'num_unique_image_xrefs': features.get('num_unique_image_xrefs', 0),
                            'num_junk_image_xrefs': features.get('num_junk_image_xrefs', 0)
                        }
                        self.results['doc_features'].append(doc_feature_summary)

                        self.results['feature_extraction_success'] += 1

                    except Exception as e:
                        self.results['errors'][f'flatten_error_{str(e)[:50]}'] += 1
                        self.results['feature_extraction_failures'] += 1

                else:
                    self.results['errors']['empty_features'] += 1
                    self.results['feature_extraction_failures'] += 1

                pymupdf_doc.close()

            except Exception as e:
                self.results['errors'][f'extraction_error_{str(e)[:50]}'] += 1
                self.results['feature_extraction_failures'] += 1

            processing_time = time.time() - start_time
            self.results['processing_times'].append(processing_time)

            # Print progress every 50 PDFs
            if file_pdf_count % 50 == 0:
                elapsed = time.time() - file_start_time
                rate = file_pdf_count / elapsed
                logger.info(f"  Processed {file_pdf_count} PDFs ({rate:.1f} PDFs/sec)")

        logger.info(f"  Completed {warc_file}: {file_pdf_count} PDFs in {time.time() - file_start_time:.1f}s")

    def _generate_report(self):
        """Generate comprehensive analysis report."""
        logger.info("PDF FEATURE ANALYSIS REPORT")

        # Basic statistics
        logger.info(f"\nBasic Statistics:")
        logger.info(f"  Total PDFs found: {self.results['pdf_count']}")
        logger.info(f"  Valid PDFs (with header): {self.results['valid_pdf_count']}")
        logger.info(f"  Feature extraction success: {self.results['feature_extraction_success']}")
        logger.info(f"  Feature extraction failures: {self.results['feature_extraction_failures']}")
        logger.info(f"  Success rate: {self.results['feature_extraction_success']/max(1, self.results['valid_pdf_count']):.1%}")

        # Processing performance
        if self.results['processing_times']:
            logger.info(f"\nProcessing Performance:")
            times = np.array(self.results['processing_times'])
            logger.info(f"  Average time per PDF: {np.mean(times):.3f}s")
            logger.info(f"  Median time per PDF: {np.median(times):.3f}s")
            logger.info(f"  95th percentile time: {np.percentile(times, 95):.3f}s")
            logger.info(f"  Total processing time: {np.sum(times):.1f}s")

        # Truncation analysis
        logger.info(f"\nTruncation Analysis:")
        total_valid = sum(self.results['truncation_stats'].values())
        for reason, count in self.results['truncation_stats'].items():
            percentage = count / max(1, total_valid) * 100
            logger.info(f"  {reason}: {count} ({percentage:.1f}%)")

        # Page count distribution
        if self.results['page_counts']:
            logger.info(f"\nPage Count Distribution:")
            pages = np.array(self.results['page_counts'])
            logger.info(f"  Mean pages: {np.mean(pages):.1f}")
            logger.info(f"  Median pages: {np.median(pages):.1f}")
            logger.info(f"  Min/Max pages: {np.min(pages)}/{np.max(pages)}")
            logger.info(f"  Pages <= 8: {np.sum(pages <= 8)} ({np.sum(pages <= 8)/len(pages):.1%})")

            # Show distribution
            page_counts = Counter(pages)
            logger.info(f"  Page distribution (top 10):")
            for pages_num, count in page_counts.most_common(10):
                percentage = count / len(self.results['page_counts']) * 100
                logger.info(f"    {pages_num} pages: {count} PDFs ({percentage:.1f}%)")

        # Feature dimension analysis
        if self.results['feature_dimensions']:
            logger.info(f"\nFeature Dimension Analysis:")
            dims = np.array(self.results['feature_dimensions'])
            dim_counts = Counter(dims)
            logger.info(f"  Feature dimensions found: {sorted(set(dims))}")
            for dim, count in dim_counts.most_common():
                percentage = count / len(dims) * 100
                logger.info(f"    {dim} features: {count} PDFs ({percentage:.1f}%)")

        # Document characteristics
        if self.results['doc_features']:
            logger.info(f"\nDocument Characteristics:")
            df = pd.DataFrame(self.results['doc_features'])

            logger.info(f"  Form PDFs: {df['is_form'].sum()} ({df['is_form'].mean():.1%})")
            logger.info(f"  Scanner-created PDFs: {df['creator_or_producer_is_known_scanner'].sum()} ({df['creator_or_producer_is_known_scanner'].mean():.1%})")
            logger.info(f"  Average garbled text ratio: {df['garbled_text_ratio'].mean():.3f}")
            logger.info(f"  PDFs with images: {(df['num_unique_image_xrefs'] > 0).sum()} ({(df['num_unique_image_xrefs'] > 0).mean():.1%})")
            logger.info(f"  PDFs with junk images: {(df['num_junk_image_xrefs'] > 0).sum()} ({(df['num_junk_image_xrefs'] > 0).mean():.1%})")

        # Error analysis
        if self.results['errors']:
            logger.info(f"\nError Analysis:")
            total_errors = sum(self.results['errors'].values())
            sorted_errors = sorted(self.results['errors'].items(), key=lambda x: x[1], reverse=True)
            for error, count in sorted_errors:
                percentage = count / total_errors * 100
                logger.info(f"  {error}: {count} ({percentage:.1f}%)")

        # Recommendations
        logger.info(f"\nRecommendations:")

        if self.results['feature_dimensions']:
            most_common_dim = Counter(self.results['feature_dimensions']).most_common(1)[0][0]
            logger.info(f"  - Most common feature dimension: {most_common_dim}")
            logger.info(f"  - Consider training XGBoost model with {most_common_dim} features")

        if self.results['page_counts']:
            pages_le_8 = np.sum(np.array(self.results['page_counts']) <= 8)
            total_pages = len(self.results['page_counts'])
            if pages_le_8 / total_pages > 0.8:
                logger.info(f"  - {pages_le_8/total_pages:.1%} of PDFs have ≤8 pages, padding strategy needed")
            else:
                logger.info(f"  - Page distribution is varied, consider dynamic feature extraction")

        success_rate = self.results['feature_extraction_success'] / max(1, self.results['valid_pdf_count'])
        if success_rate < 0.9:
            logger.info(f"  - Feature extraction success rate is {success_rate:.1%}, investigate common errors")

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_model_file.name):
            os.unlink(self.temp_model_file.name)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze PDF features from CommonCrawl WARC files')
    parser.add_argument('--limit', type=int, help='Limit PDFs per WARC file for testing')
    parser.add_argument('--data-folder', default='spec/phase3/data', help='Folder containing WARC files')

    args = parser.parse_args()

    logger.info("PDF Feature Analysis Tool")

    analyzer = PDFFeatureAnalyzer(
        data_folder=args.data_folder,
        limit_per_file=args.limit
    )

    try:
        analyzer.analyze_warc_files()
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()