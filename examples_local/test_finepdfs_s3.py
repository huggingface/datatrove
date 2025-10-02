#!/usr/bin/env python3
"""
Test FinePDFs Pipeline with CommonCrawl S3 WARC Files

Tests the complete pipeline using publicly accessible CommonCrawl S3 data:
- Stage 1: PDFWarcReader → PDFTruncationFilter → PDFRouter → Save
- Stage 2: Docling extraction for low OCR PDFs
- Stage 3: RolmOCR extraction for high OCR PDFs

Uses the same WARC file as test_finepdfs_warc.py but from S3 instead of local.
No AWS authentication required - CommonCrawl is a public dataset.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

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
# Using CC-MAIN-2018-17 dump (same as our local WARC files)
# Segment timestamp: 1524125937193.1
# File: CC-MAIN-20180420081400-20180420101400-00000.warc.gz
WARC_DATA_FOLDER = "s3://commoncrawl/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/warc/"
WARC_PATTERN = "CC-MAIN-20180420081400-20180420101400-00000.warc.gz"  # Specific file

MODEL_PATH = "examples_local/pdf_classifier_real_data.xgb"
CLASSIFIED_OUTPUT = "examples_local/output/finepdfs_s3/classified"
TEXT_EXTRACTION_OUTPUT = "examples_local/output/finepdfs_s3/text_extraction"
OCR_EXTRACTION_OUTPUT = "examples_local/output/finepdfs_s3/ocr_extraction"
LOGGING_DIR = "examples_local/logs/finepdfs_s3"

OCR_THRESHOLD = 0.5


def test_finepdfs_s3():
    """Test full FinePDFs pipeline with CommonCrawl S3 WARC files."""

    print("=" * 80)
    print("FinePDFs Pipeline Test - CommonCrawl S3")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  S3 Bucket: s3://commoncrawl (public dataset)")
    print(f"  WARC File: {WARC_PATTERN}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  OCR Threshold: {OCR_THRESHOLD}")
    print(f"  Output: {CLASSIFIED_OUTPUT}")
    print()

    # ========================================================================
    # Stage 1: Classification - Extract PDFs from S3 WARCs and route
    # ========================================================================
    print("Stage 1: PDF Extraction from S3, Truncation Detection, and Classification")
    print("-" * 80)

    stage1_classification = LocalPipelineExecutor(
        job_name="pdf_s3_classification",
        pipeline=[
            # Extract PDFs from S3 WARC files (anonymous/public access)
            # Use paths_file to avoid directory listing (which CommonCrawl S3 doesn't allow)
            PDFWarcReader(
                data_folder=("s3://commoncrawl", {"anon": True}),  # Tuple: (path, storage_options)
                paths_file="examples_local/data/cc_warc_paths.txt",  # Explicit file paths
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

    # ========================================================================
    # Stage 2: Text Extraction Path - Low OCR probability PDFs
    # ========================================================================
    print("\nStage 2: Text Extraction (Low OCR Probability - Docling)")
    print("-" * 80)

    stage2_text_extraction = LocalPipelineExecutor(
        job_name="pdf_text_extraction",
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
        logging_dir=f"{LOGGING_DIR}/text_extraction",
        depends=stage1_classification  # Wait for classification to complete
    )

    # ========================================================================
    # Stage 3: OCR Extraction Path - High OCR probability PDFs
    # ========================================================================
    print("\nStage 3: OCR Extraction (High OCR Probability - RolmOCR)")
    print("-" * 80)

    stage3_ocr_extraction = LocalPipelineExecutor(
        job_name="pdf_ocr_extraction",
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
        logging_dir=f"{LOGGING_DIR}/ocr_extraction",
        depends=stage1_classification  # Wait for classification to complete
    )

    # ========================================================================
    # Execute Pipeline
    # ========================================================================
    print("\n" + "=" * 80)
    print("Starting Pipeline Execution")
    print("=" * 80)
    print("\nStage 1 will run first (classification)")
    print("Stages 2 & 3 will run in parallel after Stage 1 completes\n")

    try:
        # Run stage 3 (which depends on stage 1)
        # Stage 2 and 3 will automatically wait for stage 1
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
