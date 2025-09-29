#!/usr/bin/env python3
"""
Test Docling extraction on low OCR probability PDFs.

This script:
1. Loads sample PDFs with low OCR probability (text-extractable)
2. Tests Docling extraction
3. Shows sample outputs
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
from datatrove.data import Document
from datatrove.pipeline.media.extractors.extractors import DoclingExtractor


def test_docling_extraction():
    """Test Docling extraction on text-based PDFs."""
    print("Testing Docling extraction on low OCR probability PDFs...")

    # Path to sample PDFs
    sample_dir = Path("threshold_analysis/samples/very_low_ocr")
    if not sample_dir.exists():
        print(f"Sample directory not found: {sample_dir}")
        return

    # Load sample info
    sample_info_path = sample_dir / "sample_info.json"
    if not sample_info_path.exists():
        print(f"Sample info not found: {sample_info_path}")
        return

    with open(sample_info_path) as f:
        sample_info = json.load(f)

    print(f"Found {len(sample_info)} sample PDFs")

    # Initialize Docling extractor
    try:
        extractor = DoclingExtractor(timeout=60)
        print("✅ DoclingExtractor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DoclingExtractor: {e}")
        return

    # Test extraction on first 2 PDFs
    for i, pdf_info in enumerate(sample_info[:2]):
        print(f"\n--- Testing PDF {i+1}: {pdf_info['id']} ---")
        print(f"OCR probability: {pdf_info['ocr_prob']:.3f}")
        print(f"Pages: {pdf_info['num_pages']}")
        print(f"Is form: {pdf_info['is_form']}")

        # Load PDF bytes
        pdf_path = sample_dir / pdf_info['saved_filename']
        if not pdf_path.exists():
            print(f"❌ PDF file not found: {pdf_path}")
            continue

        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        print(f"PDF size: {len(pdf_bytes)} bytes")

        # Create Document
        doc = Document(
            text=pdf_bytes,
            id=pdf_info['id'],
            metadata={
                'url': pdf_info.get('url', ''),
                'content_length': len(pdf_bytes),
                'ocr_prob': pdf_info['ocr_prob']
            }
        )

        # Test extraction
        try:
            print("🔄 Running Docling extraction...")
            extracted_text, metadata = extractor.extract(pdf_bytes)

            print(f"✅ Extraction successful!")
            print(f"Extracted text length: {len(extracted_text)} characters")
            print(f"Metadata keys: {list(metadata.keys())}")

            # Show first 200 characters
            if extracted_text:
                preview = extracted_text[:200].replace('\n', ' ')
                print(f"Text preview: {preview}...")
            else:
                print("⚠️  No text extracted")

        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_docling_extraction()