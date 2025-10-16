#!/usr/bin/env python3
"""
Example 08b: PDF Classifier Training

Trains XGBoost PDF classifier on real CommonCrawl data to predict OCR needs.

Components:
- PDFWarcReader: Read PDFs from WARC files
- PDFFeatureExtractor: Extract PDF features for classification
- XGBClassifier: Train binary classifier for OCR prediction

Usage:
    python spec/phase3/examples/08b_pdf_classifier_training.py --max-pdfs 1000 --data-folder data
"""

import io
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from datatrove.pipeline.media.predictor.scanned_pdf_predictor import (
    PDFFeatureExtractor,
    flatten_per_page_features,
)
from datatrove.pipeline.readers.pdf_warc import PDFWarcReader
from datatrove.utils.logging import logger

# Configuration
OUTPUT_DIR = "spec/phase3/data"


class PDFClassifierTrainer:
    def __init__(self, data_folder="data", max_pdfs=1000):
        self.data_folder = data_folder
        self.max_pdfs = max_pdfs
        self.features = []
        self.labels = []
        self.feature_names = []
        self.pdf_metadata = []

    def extract_features_and_labels(self):
        """Extract features from PDFs and create training labels."""
        logger.info("Extracting features from CommonCrawl PDFs...")

        warc_files = [f for f in os.listdir(self.data_folder) if f.endswith('.warc.gz')]
        feature_extractor = PDFFeatureExtractor(num_pages_to_sample=8)

        processed_count = 0
        success_count = 0

        for warc_file in warc_files:
            if processed_count >= self.max_pdfs:
                break

            logger.info(f"Processing {warc_file}...")
            reader = PDFWarcReader(
                data_folder=self.data_folder,
                glob_pattern=warc_file,
                limit=-1
            )

            for doc in reader.run():
                if processed_count >= self.max_pdfs:
                    break

                processed_count += 1

                # Only process valid PDFs
                if not doc.text.startswith(b'%PDF-'):
                    continue

                try:
                    import pymupdf
                    pymupdf_doc = pymupdf.open(stream=io.BytesIO(doc.text), filetype="pdf")

                    # Extract features
                    features_raw = feature_extractor.run(pymupdf_doc)
                    if not features_raw:
                        pymupdf_doc.close()
                        continue

                    features = features_raw[0]
                    flattened = flatten_per_page_features(features, sample_to_k_page_features=8)

                    # Create training label using heuristics
                    label = self._create_training_label(features, doc.metadata)

                    # Store data
                    self.features.append(list(flattened.values()))
                    self.labels.append(label)

                    if not self.feature_names:  # Store feature names once
                        self.feature_names = list(flattened.keys())

                    # Store metadata for analysis
                    self.pdf_metadata.append({
                        'id': doc.id,
                        'url': doc.metadata.get('url', ''),
                        'content_length': doc.metadata.get('content_length', 0),
                        'is_truncated': doc.metadata.get('is_truncated', False),
                        'num_pages': features.get('num_pages_successfully_sampled', 0),
                        'is_form': features.get('is_form', False),
                        'scanner_created': features.get('creator_or_producer_is_known_scanner', False),
                        'garbled_ratio': features.get('garbled_text_ratio', 0.0),
                        'predicted_needs_ocr': label
                    })

                    success_count += 1

                    if success_count % 100 == 0:
                        logger.info(f"  Extracted {success_count} successful features")

                    pymupdf_doc.close()

                except Exception as e:
                    continue

        logger.info(f"Feature extraction complete: {success_count} PDFs with features")
        return success_count

    def _create_training_label(self, features, doc_metadata):
        """
        Create training labels based on heuristics.

        Label = 1 (needs OCR) if:
        - Created by scanner software
        - High proportion of images/bitmap content
        - Very low text content
        - Form PDF with complex layout
        - High garbled text ratio

        Label = 0 (text extraction) otherwise
        """
        score = 0.0

        # Scanner software detection (strong indicator)
        if features.get('creator_or_producer_is_known_scanner', False):
            score += 0.4

        # Form PDFs often need OCR
        if features.get('is_form', False):
            score += 0.2

        # High garbled text ratio
        garbled_ratio = features.get('garbled_text_ratio', 0.0)
        if garbled_ratio > 0.1:
            score += 0.3
        elif garbled_ratio > 0.05:
            score += 0.1

        # Image-heavy documents
        total_images = sum(features.get('page_level_image_counts', [0]))
        total_pages = features.get('num_pages_successfully_sampled', 1)
        images_per_page = total_images / max(1, total_pages)

        if images_per_page > 2:
            score += 0.2
        elif images_per_page > 1:
            score += 0.1

        # High bitmap coverage
        max_bitmap_coverage = max(features.get('page_level_bitmap_proportions', [0]))
        if max_bitmap_coverage > 0.5:
            score += 0.3
        elif max_bitmap_coverage > 0.2:
            score += 0.1

        # Low text content (may indicate scanned document)
        avg_char_count = np.mean(features.get('page_level_char_counts', [0]))
        if avg_char_count < 100:
            score += 0.2
        elif avg_char_count < 500:
            score += 0.1

        # Truncated documents more likely to need OCR
        if doc_metadata.get('is_truncated', False):
            score += 0.1

        # Convert score to binary label (threshold = 0.5)
        return 1 if score >= 0.5 else 0

    def train_model(self):
        """Train XGBoost classifier on extracted features."""
        if not self.features:
            raise ValueError("No features extracted. Run extract_features_and_labels() first.")

        logger.info(f"Training XGBoost classifier on {len(self.features)} PDFs...")

        # Convert to numpy arrays
        X = np.array(self.features)
        y = np.array(self.labels)

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Label distribution: {np.bincount(y)} (0=text, 1=OCR)")
        logger.info(f"OCR percentage: {np.mean(y):.1%}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train XGBoost model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        logger.info("Training model...")
        start_time = time.time()

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f}s")

        # Evaluate model
        self._evaluate_model(model, X_test, y_test)

        # Save model and metadata
        self._save_model(model)

        return model

    def _evaluate_model(self, model, X_test, y_test):
        """Evaluate trained model performance."""
        logger.info("Model Evaluation:")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Classification report
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Text', 'OCR'])}")

        # Confusion matrix
        logger.info("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"           Predicted")
        logger.info(f"         Text   OCR")
        logger.info(f"Actual Text {cm[0,0]:4d} {cm[0,1]:4d}")
        logger.info(f"       OCR  {cm[1,0]:4d} {cm[1,1]:4d}")

        # ROC AUC
        auc = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"ROC AUC Score: {auc:.3f}")

        # Feature importance
        logger.info("Top 10 Most Important Features:")
        feature_importance = model.feature_importances_
        indices = np.argsort(feature_importance)[::-1][:10]

        for i, idx in enumerate(indices):
            importance = feature_importance[idx]
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            logger.info(f"  {i+1:2d}. {feature_name}: {importance:.4f}")

    def _save_model(self, model):
        """Save trained model and metadata."""
        model_path = OUTPUT_DIR + "/pdf_classifier_real_data.xgb"
        metadata_path = OUTPUT_DIR + "/pdf_classifier_metadata.pkl"

        # Save XGBoost model
        model.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")

        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'training_samples': len(self.features),
            'ocr_percentage': np.mean(self.labels),
            'pdf_metadata': self.pdf_metadata
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Metadata saved to: {metadata_path}")

    def analyze_training_data(self):
        """Analyze the training data characteristics."""
        if not self.pdf_metadata:
            logger.info("No metadata available for analysis")
            return

        logger.info("Training Data Analysis:")

        df = pd.DataFrame(self.pdf_metadata)

        logger.info(f"Total PDFs: {len(df)}")
        logger.info(f"OCR label distribution:")
        logger.info(f"  Text-based (0): {(df['predicted_needs_ocr'] == 0).sum()} ({(df['predicted_needs_ocr'] == 0).mean():.1%})")
        logger.info(f"  OCR-based (1): {(df['predicted_needs_ocr'] == 1).sum()} ({(df['predicted_needs_ocr'] == 1).mean():.1%})")

        logger.info(f"Document characteristics by label:")
        for label, name in [(0, 'Text-based'), (1, 'OCR-based')]:
            subset = df[df['predicted_needs_ocr'] == label]
            if len(subset) == 0:
                continue
            logger.info(f"  {name} PDFs ({len(subset)}):")
            logger.info(f"    Avg pages: {subset['num_pages'].mean():.1f}")
            logger.info(f"    Form PDFs: {subset['is_form'].mean():.1%}")
            logger.info(f"    Scanner-created: {subset['scanner_created'].mean():.1%}")
            logger.info(f"    Avg garbled ratio: {subset['garbled_ratio'].mean():.3f}")
            logger.info(f"    Truncated: {subset['is_truncated'].mean():.1%}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train PDF classifier on real data')
    parser.add_argument('--max-pdfs', type=int, default=1000, help='Maximum PDFs to process')
    parser.add_argument('--data-folder', default='data', help='Folder containing WARC files')

    args = parser.parse_args()

    logger.info("PDF Classifier Training")

    trainer = PDFClassifierTrainer(
        data_folder=args.data_folder,
        max_pdfs=args.max_pdfs
    )

    # Extract features and create labels
    success_count = trainer.extract_features_and_labels()

    if success_count < 10:
        logger.error("Too few successful feature extractions for training")
        return

    # Analyze training data
    trainer.analyze_training_data()

    # Train model
    model = trainer.train_model()

    logger.info("Training complete! Model ready for use in PDF processing pipeline.")


if __name__ == "__main__":
    main()