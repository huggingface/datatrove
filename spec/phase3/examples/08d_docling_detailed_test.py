#!/usr/bin/env python3
"""
Test Docling extraction using LocalPipelineExecutor pattern.

This script follows the same approach as Docling-sync's run_extract.py
to avoid mutex/threading issues with DoclingExtractor.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.media.extractors.extractors import DoclingExtractor
from datatrove.pipeline.media.readers.warc_threaded import WarcReaderFast
from datatrove.executor.local import LocalPipelineExecutor


def create_test_pipeline(test_data_file: str, output_dir: str, samples: int = 1):
    """Create pipeline using LocalPipelineExecutor pattern from Docling-sync."""

    # Initialize WarcReaderFast and DoclingExtractor
    reader = WarcReaderFast(data_folder="s3://commoncrawl", workers=1)
    extractor = DoclingExtractor(timeout=2*60)

    pipeline = [
        JsonlReader(
            data_folder=str(Path(test_data_file).parent),
            glob_pattern=Path(test_data_file).name,
            limit=samples,
            shuffle_paths=False,
            doc_progress=True,
        ),
        reader,
        extractor,
        JsonlWriter(output_dir)
    ]

    return pipeline


def test_docling_with_datatrove_data():
    """Test DoclingExtractor using Docling-sync data file."""
    print("Testing DoclingExtractor with LocalPipelineExecutor pattern...")

    # Use fixed JSONL format for testing
    test_data_file = "spec/phase3/data/test_sample_fixed.jsonl.gz"

    if not Path(test_data_file).exists():
        print(f"❌ Test data file not found: {test_data_file}")
        print("Please ensure Docling-sync repo is available with test data.")
        return

    print(f"✅ Using test data: {test_data_file}")

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "docling_test_output")

        try:
            # Create pipeline (process only 1 record for testing)
            pipeline = create_test_pipeline(test_data_file, output_dir, samples=1)
            print("✅ Pipeline created successfully")

            # Run with LocalPipelineExecutor (tasks=1 to avoid mutex issues)
            executor = LocalPipelineExecutor(pipeline, tasks=1)
            print("🔄 Running DoclingExtractor via LocalPipelineExecutor...")

            executor.run()

            print("✅ Pipeline execution completed!")

            # Check output
            output_files = list(Path(output_dir).glob("*.jsonl*"))
            if output_files:
                print(f"📄 Output files generated: {len(output_files)}")

                # Read and preview first output file
                output_file = output_files[0]
                print(f"📖 Reading output from: {output_file}")

                with open(output_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 1:  # Show first result only
                            break
                        data = json.loads(line)

                        print(f"\n--- Document {i+1} ---")
                        print(f"ID: {data.get('id', 'N/A')}")
                        print(f"Text length: {len(data.get('text', ''))} characters")

                        # Show text preview
                        text = data.get('text', '')
                        if text:
                            preview = text[:200].replace('\n', ' ')
                            print(f"Text preview: {preview}...")
                        else:
                            print("⚠️  No text content found")

                        # Show metadata
                        metadata = data.get('metadata', {})
                        print(f"Metadata keys: {list(metadata.keys())}")

            else:
                print("⚠️  No output files generated")

        except Exception as e:
            print(f"❌ Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()


def test_docling_with_local_pdfs():
    """Test DoclingExtractor with local PDF samples (if available)."""
    print("\nTesting DoclingExtractor with local PDF samples...")

    # Path to local PDF samples
    sample_dir = Path("spec/phase3/threshold_analysis/samples/very_low_ocr")
    if not sample_dir.exists():
        print(f"⚠️  Local PDF samples not found at: {sample_dir}")
        print("Skipping local PDF test.")
        return

    # Load sample info
    sample_info_path = sample_dir / "sample_info.json"
    if not sample_info_path.exists():
        print(f"⚠️  Sample info not found: {sample_info_path}")
        return

    print(f"✅ Found local PDF samples at: {sample_dir}")

    with open(sample_info_path) as f:
        sample_info = json.load(f)

    print(f"📂 Found {len(sample_info)} sample PDFs")

    # TODO: Implement LocalPipelineExecutor approach for local PDFs
    # This would require creating temporary JSONL files with PDF data
    # For now, just show that we found the samples

    for i, pdf_info in enumerate(sample_info[:2]):
        print(f"  - PDF {i+1}: {pdf_info['id']} (OCR prob: {pdf_info['ocr_prob']:.3f})")


if __name__ == "__main__":
    # Test with Docling-sync data (proven to work)
    test_docling_with_datatrove_data()

    # Test with local PDFs (if available)
    test_docling_with_local_pdfs()