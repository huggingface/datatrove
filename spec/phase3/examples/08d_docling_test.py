#!/usr/bin/env python3
"""
Example 08d: Docling Extractor Test

Tests DoclingExtractor on local PDF files.

Components:
- DoclingExtractor: Extract text from PDFs using Docling
- JsonlWriter: Save extracted text

Usage:
    python spec/phase3/examples/08d_docling_test.py
"""

import gzip
import json
import tempfile
from pathlib import Path

from datatrove.data import Document, Media, MediaType
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.media.extractors.extractors import DoclingExtractor
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.logging import logger

# Configuration
OUTPUT_DIR = "spec/phase3/output/docling_test"
LOGS_DIR = "spec/phase3/logs/docling_test"


def test_local_pdf_extraction():
    """Test DoclingExtractor directly on local PDF files."""

    logger.info("Testing DoclingExtractor on local PDF files...")

    # Path to local PDF samples
    sample_dir = Path("spec/phase3/threshold_analysis/samples/very_low_ocr")
    sample_info_path = sample_dir / "sample_info.json"

    if not sample_info_path.exists():
        logger.info(f"❌ Sample info not found: {sample_info_path}")
        return

    # Load sample info
    with open(sample_info_path) as f:
        sample_info = json.load(f)

    logger.info(f"Found {len(sample_info)} PDF samples")

    # Initialize DoclingExtractor
    try:
        extractor = DoclingExtractor(timeout=60)  # 1 minute timeout
        logger.info("✅ DoclingExtractor initialized successfully")
    except Exception as e:
        logger.info(f"❌ Failed to initialize DoclingExtractor: {e}")
        return

    # Test on first PDF only
    pdf_info = sample_info[0]
    logger.info(f"\n--- Testing PDF: {pdf_info['id']} ---")
    logger.info(f"OCR probability: {pdf_info['ocr_prob']:.3f}")
    logger.info(f"Pages: {pdf_info['num_pages']}")
    logger.info(f"Is form: {pdf_info['is_form']}")

    # Load PDF file
    pdf_path = sample_dir / pdf_info['saved_filename']
    if not pdf_path.exists():
        logger.info(f"❌ PDF file not found: {pdf_path}")
        return

    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    logger.info(f"PDF size: {len(pdf_bytes):,} bytes")

    # Create Document with PDF bytes in Media object (correct pattern)
    doc = Document(
        text="",  # Empty until extracted
        id=pdf_info['id'],
        media=[
            Media(
                id=pdf_info['id'],
                type=MediaType.DOCUMENT,
                media_bytes=pdf_bytes,  # Correct: PDF bytes in Media object
                url=f"file://{pdf_path}",
            )
        ],
        metadata={
            'content_length': len(pdf_bytes),
            'ocr_prob': pdf_info['ocr_prob'],
            'content_mime_detected': 'application/pdf'
        }
    )

    # Test extraction using full pipeline with LocalPipelineExecutor
    try:
        logger.info("🔄 Running DoclingExtractor through pipeline...")

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            # Run through LocalPipelineExecutor (tests Media objects properly)
            pipeline_executor = LocalPipelineExecutor(
                pipeline=[
                    [doc],  # Document with Media object
                    extractor,  # Will process doc.media[0].media_bytes
                    JsonlWriter(str(output_dir))  # Write extracted text
                ],
                tasks=1,
                logging_dir=None
            )

            pipeline_executor.run()

            # Read back the results
            output_file = output_dir / "00000.jsonl.gz"
            if not output_file.exists():
                logger.info("❌ No output file generated")
                return

            with gzip.open(output_file, 'rt') as f:
                result = json.loads(f.readline())

            extracted_text = result['text']
            metadata = result.get('metadata', {})

            logger.info(f"✅ Extraction successful!")
            logger.info(f"Extracted text length: {len(extracted_text):,} characters")
            logger.info(f"Returned metadata keys: {list(metadata.keys()) if metadata else 'None'}")

        # Show text preview
        if extracted_text:
            # Clean up the text for preview
            preview = extracted_text.replace('\n', ' ').replace('\r', ' ')
            preview = ' '.join(preview.split())  # Normalize whitespace
            preview = preview[:300]  # First 300 chars
            logger.info(f"\nExtracted text preview:")
            logger.info(f"'{preview}...'")

            # Show some statistics
            lines = extracted_text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            logger.info(f"\nText statistics:")
            logger.info(f"  Total lines: {len(lines)}")
            logger.info(f"  Non-empty lines: {len(non_empty_lines)}")
            logger.info(f"  Average line length: {sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0:.1f}")
        else:
            logger.info("⚠️  No text extracted")

    except Exception as e:
        logger.info(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()


def test_high_ocr_pdf_extraction():
    """Test DoclingExtractor on high OCR probability PDFs."""

    logger.info("\nTesting DoclingExtractor on high OCR probability PDF...")

    # Path to high OCR PDF samples
    sample_dir = Path("spec/phase3/threshold_analysis/samples/high_ocr")
    sample_info_path = sample_dir / "sample_info.json"

    if not sample_info_path.exists():
        logger.info(f"❌ High OCR sample info not found: {sample_info_path}")
        return

    # Load sample info
    with open(sample_info_path) as f:
        sample_info = json.load(f)

    logger.info(f"Found {len(sample_info)} high OCR PDF samples")

    # Initialize DoclingExtractor
    try:
        extractor = DoclingExtractor(timeout=60)  # 1 minute timeout
        logger.info("✅ DoclingExtractor initialized successfully")
    except Exception as e:
        logger.info(f"❌ Failed to initialize DoclingExtractor: {e}")
        return

    # Test on first high OCR PDF
    pdf_info = sample_info[0]
    logger.info(f"\n--- Testing High OCR PDF: {pdf_info['id']} ---")
    logger.info(f"OCR probability: {pdf_info['ocr_prob']:.3f}")
    logger.info(f"Pages: {pdf_info['num_pages']}")
    logger.info(f"Is form: {pdf_info['is_form']}")

    # Load PDF file
    pdf_path = sample_dir / pdf_info['saved_filename']
    if not pdf_path.exists():
        logger.info(f"❌ PDF file not found: {pdf_path}")
        return

    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    logger.info(f"PDF size: {len(pdf_bytes):,} bytes")

    # Create Document with PDF bytes in Media object (correct pattern)
    doc = Document(
        text="",  # Empty until extracted
        id=pdf_info['id'],
        media=[
            Media(
                id=pdf_info['id'],
                type=MediaType.DOCUMENT,
                media_bytes=pdf_bytes,  # Correct: PDF bytes in Media object
                url=f"file://{pdf_path}",
            )
        ],
        metadata={
            'content_length': len(pdf_bytes),
            'ocr_prob': pdf_info['ocr_prob'],
            'content_mime_detected': 'application/pdf'
        }
    )

    # Test extraction using full pipeline with LocalPipelineExecutor
    try:
        logger.info("🔄 Running DoclingExtractor through pipeline...")

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            # Run through LocalPipelineExecutor (tests Media objects properly)
            pipeline_executor = LocalPipelineExecutor(
                pipeline=[
                    [doc],  # Document with Media object
                    extractor,  # Will process doc.media[0].media_bytes
                    JsonlWriter(str(output_dir))  # Write extracted text
                ],
                tasks=1,
                logging_dir=None
            )

            pipeline_executor.run()

            # Read back the results
            output_file = output_dir / "00000.jsonl.gz"
            if not output_file.exists():
                logger.info("❌ No output file generated")
                return

            with gzip.open(output_file, 'rt') as f:
                result = json.loads(f.readline())

            extracted_text = result['text']
            metadata = result.get('metadata', {})

            logger.info(f"✅ Extraction successful!")
            logger.info(f"Extracted text length: {len(extracted_text):,} characters")
            logger.info(f"Returned metadata keys: {list(metadata.keys()) if metadata else 'None'}")

        # Show text preview
        if extracted_text:
            preview = extracted_text.replace('\n', ' ').replace('\r', ' ')
            preview = ' '.join(preview.split())  # Normalize whitespace
            preview = preview[:300]  # First 300 chars
            logger.info(f"\nExtracted text preview:")
            logger.info(f"'{preview}...'")

            # Show some statistics
            lines = extracted_text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            logger.info(f"\nText statistics:")
            logger.info(f"  Total lines: {len(lines)}")
            logger.info(f"  Non-empty lines: {len(non_empty_lines)}")
            logger.info(f"  Average line length: {sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0:.1f}")
        else:
            logger.info("⚠️  No text extracted")

    except Exception as e:
        logger.info(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()


def compare_ocr_thresholds():
    """Compare DoclingExtractor performance across different OCR probability thresholds."""

    logger.info("COMPARING DOCLING EXTRACTOR ACROSS OCR THRESHOLDS")

    # Test categories in order of OCR probability
    categories = [
        ("very_low_ocr", "Very Low OCR"),
        ("low_ocr", "Low OCR"),
        ("medium_ocr", "Medium OCR"),
        ("high_ocr", "High OCR"),
        ("very_high_ocr", "Very High OCR")
    ]

    for category, description in categories:
        logger.info(f"\n--- {description} ---")

        sample_dir = Path(f"spec/phase3/threshold_analysis/samples/{category}")
        sample_info_path = sample_dir / "sample_info.json"

        if not sample_info_path.exists():
            logger.info(f"⚠️  No samples found for {category}")
            continue

        with open(sample_info_path) as f:
            sample_info = json.load(f)

        if not sample_info:
            logger.info(f"⚠️  No sample data for {category}")
            continue

        # Test first sample from each category
        pdf_info = sample_info[0]
        logger.info(f"Sample: {pdf_info['id']}")
        logger.info(f"OCR probability: {pdf_info['ocr_prob']:.3f}")
        logger.info(f"Pages: {pdf_info['num_pages']}")


if __name__ == "__main__":
    # Test very low OCR probability PDF
    test_local_pdf_extraction()

    # Test high OCR probability PDF
    test_high_ocr_pdf_extraction()

    # Compare DoclingExtractor across all thresholds
    compare_ocr_thresholds()