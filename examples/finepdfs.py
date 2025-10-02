#!/usr/bin/env python3
"""
Two-Tiered PDF Processing Pipeline (FinePDFs Reproduction)

Implements intelligent routing of PDFs based on XGBoost classifier predictions:
- Low OCR probability PDFs → Docling for direct text extraction
- High OCR probability PDFs → RolmOCR for GPU-based OCR

Architecture: Three-stage pipeline with direct WARC streaming:
1. Classification: Extract PDFs from WARCs, filter truncated, route by OCR probability
2. Text Extraction Path: Process low OCR PDFs with Docling
3. OCR Extraction Path: Process high OCR PDFs with RolmOCR
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.readers.pdf_warc import PDFWarcReader
from datatrove.pipeline.filters.pdf_truncation import PDFTruncationFilter
from datatrove.pipeline.filters.pdf_router import PDFRouter
from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from datatrove.pipeline.media.extractors.extractors import DoclingExtractor
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceConfig
from datatrove.pipeline.inference.post_process import ExtractInferenceText
from datatrove.pipeline.inference.query_builders.vision import rolmocr_query_builder
from datatrove.pipeline.writers.jsonl import JsonlWriter, PersistentContextJsonlWriter


# ============================================================================
# Configuration
# ============================================================================

# WARC data source (S3 or local)
# For CommonCrawl: "s3://commoncrawl/crawl-data/CC-MAIN-YYYY-WW/segments/"
WARC_DATA_FOLDER = "s3://commoncrawl/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/warc/"
WARC_GLOB_PATTERN = "*.warc.gz"  # or "CC-MAIN-*.warc.gz" for specific files

# XGBoost model path
MODEL_PATH = "examples_local/pdf_classifier_real_data.xgb"

# Output paths
CLASSIFIED_OUTPUT = "output/finepdfs/classified"
TEXT_EXTRACTION_OUTPUT = "output/finepdfs/text_extraction"
OCR_EXTRACTION_OUTPUT = "output/finepdfs/ocr_extraction"

# Logging
LOGGING_DIR = "logs/finepdfs"

# Processing configuration
OCR_THRESHOLD = 0.5
PDF_LIMIT = -1  # Set to limit number of PDFs processed (-1 for all)
NUM_TASKS = 1  # Number of parallel tasks


# ============================================================================
# Pipeline Definition
# ============================================================================

def create_production_pipeline():
    """Create the three-stage production pipeline with direct WARC streaming."""

    print("=" * 80)
    print("FinePDFs Production Pipeline - Two-Tiered PDF Processing")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  WARC Data: {WARC_DATA_FOLDER}")
    print(f"  WARC Pattern: {WARC_GLOB_PATTERN}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  OCR Threshold: {OCR_THRESHOLD}")
    print(f"  PDF Limit: {PDF_LIMIT if PDF_LIMIT > 0 else 'All'}")
    print(f"  Output: {CLASSIFIED_OUTPUT}")
    print()

    # ========================================================================
    # Stage 1: Classification - Extract PDFs from WARCs and route
    # ========================================================================
    print("Stage 1: PDF Extraction, Truncation Detection, and Classification")
    print("-" * 80)

    stage1_classification = LocalPipelineExecutor(
        pipeline=[
            # Extract PDFs from WARC files (streams directly)
            PDFWarcReader(
                data_folder=WARC_DATA_FOLDER,
                glob_pattern=WARC_GLOB_PATTERN,
                limit=PDF_LIMIT
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
        tasks=NUM_TASKS,
        logging_dir=f"{LOGGING_DIR}/classification"
    )

    # Run stage 1 first
    stage1_classification.run()

    # ========================================================================
    # Stage 2: Text Extraction Path - Low OCR probability PDFs
    # ========================================================================
    print("\nStage 2: Text Extraction (Low OCR Probability - Docling)")
    print("-" * 80)

    stage2_text_extraction = LocalPipelineExecutor(
        pipeline=[
            # Read pre-classified PDFs with Media objects
            JsonlReader(CLASSIFIED_OUTPUT, glob_pattern="*.jsonl.gz"),
            # Filter for low OCR PDFs
            LambdaFilter(
                filter_function=lambda doc: doc.metadata.get("processing_route") == "text_extraction"
            ),
            # Extract text with Docling
            DoclingExtractor(timeout=10*60),  # 10 minute timeout per PDF
            # Save extracted text
            JsonlWriter(TEXT_EXTRACTION_OUTPUT),
        ],
        tasks=NUM_TASKS,
        logging_dir=f"{LOGGING_DIR}/text_extraction"
    )

    # ========================================================================
    # Stage 3: OCR Extraction Path - High OCR probability PDFs
    # ========================================================================
    print("\nStage 3: OCR Extraction (High OCR Probability - RolmOCR)")
    print("-" * 80)

    stage3_ocr_extraction = LocalPipelineExecutor(
        pipeline=[
            # Read same pre-classified PDFs
            JsonlReader(CLASSIFIED_OUTPUT, glob_pattern="*.jsonl.gz"),
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
        tasks=NUM_TASKS,
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


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run the complete FinePDFs production pipeline."""

    # Check prerequisites
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: XGBoost model not found: {MODEL_PATH}")
        print("\nTrain the model using examples_local/08b_pdf_classifier_model.py")
        return

    # Run pipeline
    try:
        create_production_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
