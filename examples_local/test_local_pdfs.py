#!/usr/bin/env python3
"""
Test DoclingExtractor on local PDF files directly.

This bypasses the WARC/S3 complexity and tests DoclingExtractor
with actual PDF files we have locally.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
from datatrove.data import Document, Media, MediaType
from datatrove.pipeline.media.extractors.extractors import DoclingExtractor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.writers.jsonl import JsonlWriter
import tempfile
import gzip


def test_local_pdf_extraction():
    """Test DoclingExtractor directly on local PDF files."""

    print("Testing DoclingExtractor on local PDF files...")

    # Path to local PDF samples
    sample_dir = Path("examples_local/threshold_analysis/samples/very_low_ocr")
    sample_info_path = sample_dir / "sample_info.json"

    if not sample_info_path.exists():
        print(f"❌ Sample info not found: {sample_info_path}")
        return

    # Load sample info
    with open(sample_info_path) as f:
        sample_info = json.load(f)

    print(f"Found {len(sample_info)} PDF samples")

    # Initialize DoclingExtractor
    try:
        extractor = DoclingExtractor(timeout=60)  # 1 minute timeout
        print("✅ DoclingExtractor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DoclingExtractor: {e}")
        return

    # Test on first PDF only
    pdf_info = sample_info[0]
    print(f"\n--- Testing PDF: {pdf_info['id']} ---")
    print(f"OCR probability: {pdf_info['ocr_prob']:.3f}")
    print(f"Pages: {pdf_info['num_pages']}")
    print(f"Is form: {pdf_info['is_form']}")

    # Load PDF file
    pdf_path = sample_dir / pdf_info['saved_filename']
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return

    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    print(f"PDF size: {len(pdf_bytes):,} bytes")

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
        print("🔄 Running DoclingExtractor through pipeline...")

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
                print("❌ No output file generated")
                return

            with gzip.open(output_file, 'rt') as f:
                result = json.loads(f.readline())

            extracted_text = result['text']
            metadata = result.get('metadata', {})

            print(f"✅ Extraction successful!")
            print(f"Extracted text length: {len(extracted_text):,} characters")
            print(f"Returned metadata keys: {list(metadata.keys()) if metadata else 'None'}")

        # Show text preview
        if extracted_text:
            # Clean up the text for preview
            preview = extracted_text.replace('\n', ' ').replace('\r', ' ')
            preview = ' '.join(preview.split())  # Normalize whitespace
            preview = preview[:300]  # First 300 chars
            print(f"\nExtracted text preview:")
            print(f"'{preview}...'")

            # Show some statistics
            lines = extracted_text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            print(f"\nText statistics:")
            print(f"  Total lines: {len(lines)}")
            print(f"  Non-empty lines: {len(non_empty_lines)}")
            print(f"  Average line length: {sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0:.1f}")
        else:
            print("⚠️  No text extracted")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()


def test_high_ocr_pdf_extraction():
    """Test DoclingExtractor on high OCR probability PDFs."""

    print("\nTesting DoclingExtractor on high OCR probability PDF...")

    # Path to high OCR PDF samples
    sample_dir = Path("examples_local/threshold_analysis/samples/high_ocr")
    sample_info_path = sample_dir / "sample_info.json"

    if not sample_info_path.exists():
        print(f"❌ High OCR sample info not found: {sample_info_path}")
        return

    # Load sample info
    with open(sample_info_path) as f:
        sample_info = json.load(f)

    print(f"Found {len(sample_info)} high OCR PDF samples")

    # Initialize DoclingExtractor
    try:
        extractor = DoclingExtractor(timeout=60)  # 1 minute timeout
        print("✅ DoclingExtractor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DoclingExtractor: {e}")
        return

    # Test on first high OCR PDF
    pdf_info = sample_info[0]
    print(f"\n--- Testing High OCR PDF: {pdf_info['id']} ---")
    print(f"OCR probability: {pdf_info['ocr_prob']:.3f}")
    print(f"Pages: {pdf_info['num_pages']}")
    print(f"Is form: {pdf_info['is_form']}")

    # Load PDF file
    pdf_path = sample_dir / pdf_info['saved_filename']
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return

    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    print(f"PDF size: {len(pdf_bytes):,} bytes")

    # Create Document with PDF bytes
    doc = Document(
        text=pdf_bytes,
        id=pdf_info['id'],
        metadata={
            'url': f"file://{pdf_path}",
            'content_length': len(pdf_bytes),
            'ocr_prob': pdf_info['ocr_prob'],
            'content_mime_detected': 'application/pdf'
        }
    )

    # Test extraction
    try:
        print("🔄 Running DoclingExtractor on high OCR probability PDF...")

        extracted_text, metadata = extractor.extract((pdf_bytes, doc.metadata))

        print(f"✅ Extraction successful!")
        print(f"Extracted text length: {len(extracted_text):,} characters")
        print(f"Returned metadata keys: {list(metadata.keys()) if metadata else 'None'}")

        # Show text preview
        if extracted_text:
            preview = extracted_text.replace('\n', ' ').replace('\r', ' ')
            preview = ' '.join(preview.split())  # Normalize whitespace
            preview = preview[:300]  # First 300 chars
            print(f"\nExtracted text preview:")
            print(f"'{preview}...'")

            # Show some statistics
            lines = extracted_text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            print(f"\nText statistics:")
            print(f"  Total lines: {len(lines)}")
            print(f"  Non-empty lines: {len(non_empty_lines)}")
            print(f"  Average line length: {sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0:.1f}")
        else:
            print("⚠️  No text extracted")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()


def compare_ocr_thresholds():
    """Compare DoclingExtractor performance across different OCR probability thresholds."""

    print("\n" + "="*60)
    print("COMPARING DOCLING EXTRACTOR ACROSS OCR THRESHOLDS")
    print("="*60)

    # Test categories in order of OCR probability
    categories = [
        ("very_low_ocr", "Very Low OCR"),
        ("low_ocr", "Low OCR"),
        ("medium_ocr", "Medium OCR"),
        ("high_ocr", "High OCR"),
        ("very_high_ocr", "Very High OCR")
    ]

    for category, description in categories:
        print(f"\n--- {description} ---")

        sample_dir = Path(f"examples_local/threshold_analysis/samples/{category}")
        sample_info_path = sample_dir / "sample_info.json"

        if not sample_info_path.exists():
            print(f"⚠️  No samples found for {category}")
            continue

        with open(sample_info_path) as f:
            sample_info = json.load(f)

        if not sample_info:
            print(f"⚠️  No sample data for {category}")
            continue

        # Test first sample from each category
        pdf_info = sample_info[0]
        print(f"Sample: {pdf_info['id']}")
        print(f"OCR probability: {pdf_info['ocr_prob']:.3f}")
        print(f"Pages: {pdf_info['num_pages']}")


if __name__ == "__main__":
    # Test very low OCR probability PDF
    test_local_pdf_extraction()

    # Test high OCR probability PDF
    test_high_ocr_pdf_extraction()

    # Compare DoclingExtractor across all thresholds
    compare_ocr_thresholds()