#!/usr/bin/env python3
"""
Test FinePDFs Pipeline with CommonCrawl HTTPS WARC Files

Tests the complete pipeline using publicly accessible CommonCrawl HTTPS data:
- Downloads warc.paths.gz for specified dump
- Converts to HTTPS URLs (no credentials required)
- Stage 1: PDFWarcReader → PDFTruncationFilter → PDFRouter → Save
- Stage 2: Docling extraction for low OCR PDFs
- Stage 3: RolmOCR extraction for high OCR PDFs

This is a test/development version. Production should use S3 paths with credentials
like examples/fineweb.py does.
"""

import sys
import gzip
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

import requests
from datatrove.pipeline.readers.pdf_warc import PDFWarcReader
from datatrove.pipeline.filters.pdf_truncation import PDFTruncationFilter
from datatrove.pipeline.filters.pdf_router import PDFRouter
from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from datatrove.pipeline.media.extractors.extractors import DoclingExtractor
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceConfig
from datatrove.pipeline.inference.post_process import ExtractInferenceText
from datatrove.pipeline.inference.query_builders.vision import rolmocr_query_builder
from datatrove.pipeline.writers.jsonl import JsonlWriter, PersistentContextJsonlWriter
from datatrove.pipeline.readers import JsonlReader
from datatrove.executor.local import LocalPipelineExecutor

# Configuration
DUMP_TO_PROCESS = "CC-MAIN-2018-17"  # CommonCrawl dump to process
NUM_WARC_FILES = 10  # Number of WARC files to process (for testing)

MODEL_PATH = "examples_local/pdf_classifier_real_data.xgb"
CLASSIFIED_OUTPUT = "examples_local/output/finepdfs_https/classified"
TEXT_EXTRACTION_OUTPUT = "examples_local/output/finepdfs_https/text_extraction"
OCR_EXTRACTION_OUTPUT = "examples_local/output/finepdfs_https/ocr_extraction"
LOGGING_DIR = "examples_local/logs/finepdfs_https"

OCR_THRESHOLD = 0.5


def download_warc_paths():
    """Download and convert WARC paths to HTTPS URLs."""
    paths_file = Path("examples_local/data/cc_warc_paths.txt")

    print(f"\n📥 Downloading WARC paths for {DUMP_TO_PROCESS}...")
    paths_url = f"https://data.commoncrawl.org/crawl-data/{DUMP_TO_PROCESS}/warc.paths.gz"

    print(f"   Fetching: {paths_url}")
    response = requests.get(paths_url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download paths: HTTP {response.status_code}")

    # Extract paths
    paths_content = gzip.decompress(response.content).decode('utf-8')
    warc_paths = [line.strip() for line in paths_content.split('\n') if line.strip()]

    print(f"   Found {len(warc_paths)} WARC files in dump")
    print(f"   Using first {NUM_WARC_FILES} files for testing")

    # Save relative paths (data_folder will be prepended)
    paths_file.parent.mkdir(parents=True, exist_ok=True)
    relative_paths = warc_paths[:NUM_WARC_FILES]
    paths_file.write_text('\n'.join(relative_paths))

    print(f"   ✅ Saved {len(relative_paths)} paths to {paths_file}")
    return paths_file


def test_finepdfs_s3():
    """Test full FinePDFs pipeline with CommonCrawl HTTPS WARC files."""

    print("=" * 80)
    print("FinePDFs Pipeline Test - CommonCrawl HTTPS")
    print("=" * 80)

    # Download paths file
    paths_file = download_warc_paths()

    print(f"\n⚙️  Configuration:")
    print(f"   Dump: {DUMP_TO_PROCESS}")
    print(f"   WARC files: {NUM_WARC_FILES}")
    print(f"   Paths file: {paths_file}")
    print(f"   Model: {MODEL_PATH}")
    print(f"   OCR Threshold: {OCR_THRESHOLD}")
    print(f"   Output: {CLASSIFIED_OUTPUT}")
    print()

    # ========================================================================
    # Stage 1: Classification - Extract PDFs from HTTPS WARCs and route
    # ========================================================================
    print("\n🔍 Stage 1: PDF Extraction, Truncation Detection, and Classification")
    print("-" * 80)

    stage1_classification = LocalPipelineExecutor(
        pipeline=[
            # Extract PDFs from CommonCrawl HTTPS (public access, no credentials)
            # Use paths_file with HTTPS URLs (S3 requires credentials)
            PDFWarcReader(
                data_folder="https://data.commoncrawl.org",
                paths_file="examples_local/data/cc_warc_paths.txt",  # HTTPS URLs
                limit=50  # Limit for testing
            ),
            # Filter truncated PDFs
            PDFTruncationFilter(),
            # Classify and route PDFs
            PDFRouter(
                model_path=MODEL_PATH,
                threshold=OCR_THRESHOLD
            ),
            # Save ALL PDFs with routing metadata
            JsonlWriter(CLASSIFIED_OUTPUT, save_media_bytes=True),
        ],
        tasks=1,  # Single task for testing
        logging_dir=f"{LOGGING_DIR}/classification"
    )

    # Run stage 1 first
    stage1_classification.run()

    # ========================================================================
    # Stage 2: Text Extraction Path - Low OCR probability PDFs
    # ========================================================================
    print("\n🔍 Stage 2: Text Extraction (Low OCR Probability - Docling)")
    print("-" * 80)

    stage2_text_extraction = LocalPipelineExecutor(
        pipeline=[
            # Read pre-classified PDFs with Media objects
            JsonlReader(CLASSIFIED_OUTPUT),
            # Filter for low OCR PDFs
            LambdaFilter(
                filter_function=lambda doc: doc.metadata.get("processing_route") == "text_extraction"
            ),
            # Extract text with Docling
            DoclingExtractor(timeout=10*60),  # 10 minute timeout per PDF
            # Save extracted text
            JsonlWriter(TEXT_EXTRACTION_OUTPUT),
        ],
        tasks=1,
        logging_dir=f"{LOGGING_DIR}/text_extraction"
    )

    # ========================================================================
    # Stage 3: OCR Extraction Path - High OCR probability PDFs
    # ========================================================================
    print("\n🔍 Stage 3: OCR Extraction (High OCR Probability - RolmOCR)")
    print("-" * 80)

    stage3_ocr_extraction = LocalPipelineExecutor(
        pipeline=[
            # Read same pre-classified PDFs
            JsonlReader(CLASSIFIED_OUTPUT),
            # Filter for high OCR PDFs
            LambdaFilter(
                filter_function=lambda doc: doc.metadata.get("processing_route") == "ocr_extraction"
            ),
            # Extract text with RolmOCR
            InferenceRunner(
                query_builder=rolmocr_query_builder,
                config=InferenceConfig(
                    server_type="lmdeploy",
                    model_name_or_path="Reducto/RolmOCR",
                    model_max_context=8096,
                    max_concurrent_requests=1,
                    max_concurrent_tasks=1,
                    model_kwargs={
                        "chat_template": "internlm",
                        "vision_max_batch_size": 128
                    }
                ),
                post_process_steps=[
                    ExtractInferenceText(),
                    PersistentContextJsonlWriter(OCR_EXTRACTION_OUTPUT)
                ]
            ),
        ],
        tasks=1,  # GPU-bound, single task
        logging_dir=f"{LOGGING_DIR}/ocr_extraction"
    )

    # Run stages 2 & 3
    try:
        stage2_text_extraction.run()
        stage3_ocr_extraction.run()
    finally:
        # Explicitly close the writer to ensure gzip file is properly finalized
        writer = None
        for step in stage3_ocr_extraction.pipeline:
            if isinstance(step, InferenceRunner):
                for post_step in step.post_process_steps:
                    if isinstance(post_step, PersistentContextJsonlWriter):
                        writer = post_step
                        break
        if writer and writer._context_entered:
            print("Closing OCR writer context...")
            writer.__exit__(None, None, None)

    print("\n" + "=" * 80)
    print("Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  Classified PDFs: {CLASSIFIED_OUTPUT}")
    print(f"  Text Extraction: {TEXT_EXTRACTION_OUTPUT}")
    print(f"  OCR Extraction: {OCR_EXTRACTION_OUTPUT}")
    print(f"\nLogs:")
    print(f"  {LOGGING_DIR}/*/stats/")
    print()
    print("Run extract_text_for_review.py to extract text from results")


if __name__ == "__main__":
    test_finepdfs_s3()
