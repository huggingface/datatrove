#!/usr/bin/env python3
"""
Test OCR extraction on high OCR probability PDFs.

This script:
1. Loads sample PDFs with high OCR probability (scanned/image-based)
2. Tests different OCR extraction methods
3. Compares OCR results with direct text extraction
4. Prepares for Lambda OCR server testing
"""

import os
import sys
import json
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, 'src')
from datatrove.data import Document

def test_pymupdf_ocr():
    """Test PyMuPDF OCR extraction on high OCR probability PDFs."""
    print("Testing OCR extraction on high OCR probability PDFs...")

    # Path to sample PDFs
    sample_dir = Path("examples_local/threshold_analysis/samples/high_ocr")
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

    print(f"Found {len(sample_info)} high OCR probability sample PDFs")

    # Test OCR on first 2 PDFs
    for i, pdf_info in enumerate(sample_info[:2]):
        print(f"\n--- Testing PDF {i+1}: {pdf_info['id']} ---")
        print(f"OCR probability: {pdf_info['ocr_prob']:.3f}")
        print(f"Pages: {pdf_info['num_pages']}")
        print(f"Garbled text ratio: {pdf_info['garbled_text_ratio']:.3f}")
        print(f"URL: {pdf_info['url']}")

        # Load PDF bytes
        pdf_path = sample_dir / pdf_info['saved_filename']
        if not pdf_path.exists():
            print(f"❌ PDF file not found: {pdf_path}")
            continue

        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        print(f"PDF size: {len(pdf_bytes)} bytes")

        # Test 1: Direct text extraction (should fail/be garbled)
        print("\n🔍 Testing direct text extraction...")
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            direct_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                direct_text += page.get_text()

            doc.close()

            print(f"Direct text length: {len(direct_text)} characters")
            if direct_text.strip():
                preview = direct_text.strip()[:200].replace('\n', ' ')
                print(f"Direct text preview: {preview}...")
            else:
                print("⚠️  No direct text extracted")

        except Exception as e:
            print(f"❌ Direct text extraction failed: {e}")

        # Test 2: OCR extraction
        print("\n🔄 Testing OCR extraction...")
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            ocr_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Get page as image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")

                # Try OCR with different methods
                try:
                    # Method 1: PyMuPDF OCR (if available)
                    ocr_text += page.get_textpage_ocr().extractText()
                except:
                    try:
                        # Method 2: Tesseract via pytesseract
                        import pytesseract
                        from PIL import Image
                        import io

                        img = Image.open(io.BytesIO(img_data))
                        page_text = pytesseract.image_to_string(img)
                        ocr_text += page_text

                    except ImportError:
                        print("⚠️  Neither PyMuPDF OCR nor Tesseract available")
                        break
                    except Exception as ocr_e:
                        print(f"⚠️  OCR failed for page {page_num}: {ocr_e}")

            doc.close()

            print(f"OCR text length: {len(ocr_text)} characters")
            if ocr_text.strip():
                preview = ocr_text.strip()[:200].replace('\n', ' ')
                print(f"OCR text preview: {preview}...")

                # Compare with direct text
                if len(direct_text.strip()) > 0:
                    improvement = len(ocr_text) / len(direct_text) if len(direct_text) > 0 else float('inf')
                    print(f"OCR improvement factor: {improvement:.2f}x")
            else:
                print("⚠️  No OCR text extracted")

        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
            import traceback
            traceback.print_exc()

def test_lambda_ocr_preparation():
    """Prepare data structure for Lambda OCR server testing."""
    print("\n=== Lambda OCR Preparation ===")

    sample_dir = Path("examples_local/threshold_analysis/samples/high_ocr")
    sample_info_path = sample_dir / "sample_info.json"

    if not sample_info_path.exists():
        print("No sample info found")
        return

    with open(sample_info_path) as f:
        sample_info = json.load(f)

    # Prepare Lambda payload format
    lambda_payloads = []
    for pdf_info in sample_info:
        pdf_path = sample_dir / pdf_info['saved_filename']
        if pdf_path.exists():
            payload = {
                "pdf_id": pdf_info['id'],
                "ocr_prob": pdf_info['ocr_prob'],
                "num_pages": pdf_info['num_pages'],
                "content_length": pdf_info['content_length'],
                "url": pdf_info['url'],
                "local_path": str(pdf_path)
            }
            lambda_payloads.append(payload)

    print(f"Prepared {len(lambda_payloads)} PDFs for Lambda OCR testing")

    # Save payload info
    payload_file = Path("examples_local/lambda_ocr_payloads.json")
    with open(payload_file, 'w') as f:
        json.dump(lambda_payloads, f, indent=2)

    print(f"Lambda OCR payloads saved to: {payload_file}")

    # Show example payload
    if lambda_payloads:
        print(f"\nExample Lambda payload:")
        print(json.dumps(lambda_payloads[0], indent=2))

def main():
    """Run OCR testing suite."""
    print("=== Step 4b: OCR Component Testing ===")

    # Test local OCR capabilities
    test_pymupdf_ocr()

    # Prepare for Lambda OCR testing
    test_lambda_ocr_preparation()

    print("\n✅ OCR testing setup complete!")
    print("Next steps:")
    print("- Set up Lambda OCR server (Step 4c)")
    print("- Test full pipeline with XGBoost routing (Step 4d)")

if __name__ == "__main__":
    main()