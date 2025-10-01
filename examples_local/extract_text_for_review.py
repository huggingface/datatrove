#!/usr/bin/env python3
"""
Extract text from JSONL files for easy comparison with PDFs/PNGs.

Creates .txt files with extracted text alongside metadata for manual review.
"""

import json
import gzip
from pathlib import Path


def extract_text_from_jsonl(jsonl_path: Path, output_dir: Path):
    """Extract text and metadata from JSONL to individual .txt files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📄 Processing: {jsonl_path.name}")

    # Open JSONL (handles .gz automatically)
    open_fn = gzip.open if jsonl_path.suffix == '.gz' else open

    with open_fn(jsonl_path, 'rt') as f:
        for line in f:
            doc = json.loads(line)

            doc_id = doc.get('id', 'unknown')
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})

            # Create text file with metadata header
            output_file = output_dir / f"{doc_id}.txt"

            with open(output_file, 'w') as out:
                # Header with metadata
                out.write("=" * 80 + "\n")
                out.write(f"Document ID: {doc_id}\n")
                out.write("=" * 80 + "\n\n")

                # Key metadata
                out.write("METADATA:\n")
                out.write("-" * 80 + "\n")

                # OCR probability and routing
                if 'ocr_probability' in metadata:
                    out.write(f"OCR Probability: {metadata['ocr_probability']:.4f}\n")
                if 'processing_route' in metadata:
                    out.write(f"Processing Route: {metadata['processing_route']}\n")

                # Additional useful fields
                for key in ['num_pages', 'is_form', 'garbled_text_ratio',
                           'is_encrypted', 'content_length', 'source']:
                    if key in metadata:
                        out.write(f"{key.replace('_', ' ').title()}: {metadata[key]}\n")

                out.write("-" * 80 + "\n\n")

                # Extracted text
                out.write("EXTRACTED TEXT:\n")
                out.write("=" * 80 + "\n\n")
                out.write(text)
                out.write("\n\n")
                out.write("=" * 80 + "\n")
                out.write(f"END OF DOCUMENT: {doc_id}\n")
                out.write("=" * 80 + "\n")

            print(f"  ✓ {doc_id}.txt ({len(text)} chars)")


def main():
    base_dir = Path("examples_local/output/finepdfs_local")

    print("=" * 80)
    print("Extracting Text from FinePDFs Results")
    print("=" * 80)

    # Process text extraction results (Docling - low OCR)
    text_extraction_jsonl = base_dir / "text_extraction" / "00000.jsonl.gz"
    if text_extraction_jsonl.exists():
        print("\n📚 Text Extraction Path (Docling - Low OCR)")
        extract_text_from_jsonl(
            text_extraction_jsonl,
            base_dir / "text_extraction_review"
        )

    # Process OCR extraction results (RolmOCR - high OCR)
    ocr_extraction_jsonl = base_dir / "ocr_extraction" / "00000.jsonl.gz"
    if ocr_extraction_jsonl.exists():
        print("\n🔍 OCR Extraction Path (RolmOCR - High OCR)")
        extract_text_from_jsonl(
            ocr_extraction_jsonl,
            base_dir / "ocr_extraction_review"
        )

    # Also extract classified metadata (no extracted text yet)
    classified_jsonl = base_dir / "classified" / "00000.jsonl.gz"
    if classified_jsonl.exists():
        print("\n🔀 Classification Results (Routing Metadata)")
        extract_text_from_jsonl(
            classified_jsonl,
            base_dir / "classified_review"
        )

    print("\n" + "=" * 80)
    print("✅ Text extraction complete!")
    print("=" * 80)
    print("\nReview files created:")
    print(f"  {base_dir}/text_extraction_review/")
    print(f"  {base_dir}/ocr_extraction_review/")
    print(f"  {base_dir}/classified_review/")
    print("\nCross-reference with:")
    print(f"  {base_dir}/text_extraction_pdfs/")
    print(f"  {base_dir}/ocr_extraction_pdfs/")
    print(f"  {base_dir}/ocr_extraction_pages_png/")
    print()


if __name__ == "__main__":
    main()
