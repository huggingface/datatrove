#!/usr/bin/env python3
"""
Two-Tiered PDF Processing Pipeline (FinePDFs Reproduction)

Implements intelligent routing of PDFs based on XGBoost classifier predictions:
- Low OCR probability PDFs → Docling for direct text extraction
- High OCR probability PDFs → RolmOCR for GPU-based OCR

Architecture follows FineWeb pattern with three dependent stages:
1. Classification: Route PDFs based on OCR probability
2. Text Extraction Path: Process low OCR PDFs with Docling
3. OCR Extraction Path: Process high OCR PDFs with RolmOCR
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.media.readers.warc_threaded import WarcReaderFast
from datatrove.pipeline.filters.pdf_truncation import PDFTruncationDetector
from datatrove.pipeline.filters.pdf_router import PDFRouter
from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from datatrove.pipeline.media.extractors.extractors import DoclingExtractor
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceConfig
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.run_inference import InferenceSuccess
from typing import Iterable
import fitz
import base64


# ============================================================================
# Configuration
# ============================================================================

# Input: JSONL file with WARC metadata (warc_filename, warc_record_offset)
WARC_METADATA_FILE = "data/warc_metadata.jsonl.gz"

# WARC data source (S3 or local)
WARC_DATA_FOLDER = "s3://commoncrawl"  # or local path like "data/warcs"

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
WARC_WORKERS = 4
TEXT_EXTRACTION_TASKS = 4
OCR_EXTRACTION_TASKS = 1  # GPU-bound


# ============================================================================
# RolmOCR Support Classes
# ============================================================================

class PostProcessOCRResults(PipelineStep):
    """Post-process RolmOCR inference results."""

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            # Extract OCR text from inference results and replace with extracted text
            document.text = "\n".join([
                x.text if isinstance(x, InferenceSuccess) else x.error
                for x in document.metadata["inference_results"]
            ])
            # Clean up inference_results metadata
            del document.metadata["inference_results"]
            yield document


class PersistentContextJsonlWriter(JsonlWriter):
    """JsonlWriter that keeps file context open across multiple run() calls.

    Workaround for framework bug where InferenceRunner calls post_process_steps
    separately for each document, causing JsonlWriter to close/reopen files
    between documents, which truncates the output file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context_entered = False

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        # Enter context only once, on first call
        if not self._context_entered:
            self.__enter__()
            self._context_entered = True

        # Write documents without entering/exiting context
        for document in data:
            with self.track_time():
                self.write(document, rank)
            yield document

    def __del__(self):
        # Clean up context when object is destroyed
        if self._context_entered:
            try:
                self.__exit__(None, None, None)
            except:
                pass


def rolmocr_query_builder(runner: InferenceRunner, doc: Document) -> dict:
    """Convert PDF document to RolmOCR vision request.

    Follows FinePDFs specification:
    - Rescale PDFs so longest dimension ≥ 1280px
    - Ensure representation doesn't exceed 2048 image tokens
    - Total context length set to 8096 tokens
    """
    from datatrove.pipeline.inference.utils.page_rendering import render_page_to_base64png_pymupdf

    # Get PDF bytes from Media object
    if not doc.media or not doc.media[0].media_bytes:
        raise ValueError(f"Document {doc.id} has no media bytes")

    pdf_bytes = doc.media[0].media_bytes
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Process all pages (or limit for memory)
    max_pages = len(pdf_doc)  # Process all pages in production
    page_images = []

    for page_num in range(max_pages):
        page = pdf_doc.load_page(page_num)

        # Use FinePDFs specification resolution
        base64_image = render_page_to_base64png_pymupdf(
            page,
            resize_longest_side_pixels=1280,  # FinePDFs spec
            max_visual_tokens=2048  # FinePDFs spec
        )

        page_images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })

    pdf_doc.close()

    # Create OpenAI-compatible vision request
    return {
        "model": runner.config.model_name_or_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this PDF document using OCR. Return only the extracted text."},
                    *page_images
                ]
            }
        ],
        "max_tokens": 4096,  # Leave room for input in 8096 total context
        "temperature": 0.0
    }


# ============================================================================
# Pipeline Definition
# ============================================================================

def create_production_pipeline():
    """Create the three-stage production pipeline."""

    print("=" * 80)
    print("FinePDFs Production Pipeline - Two-Tiered PDF Processing")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  WARC Metadata: {WARC_METADATA_FILE}")
    print(f"  WARC Data: {WARC_DATA_FOLDER}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  OCR Threshold: {OCR_THRESHOLD}")
    print(f"  Output: {CLASSIFIED_OUTPUT}")
    print()

    # ========================================================================
    # Stage 1: Classification - Route PDFs based on OCR probability
    # ========================================================================
    print("Stage 1: PDF Classification and Routing")
    print("-" * 80)

    stage1_classification = LocalPipelineExecutor(
        job_name="pdf_classification",
        pipeline=[
            # Read JSONL with WARC metadata (warc_filename, warc_record_offset)
            JsonlReader(
                data_folder=str(Path(WARC_METADATA_FILE).parent),
                glob_pattern=Path(WARC_METADATA_FILE).name,
            ),
            # Fetch PDFs from WARCs into doc.media
            WarcReaderFast(
                data_folder=WARC_DATA_FOLDER,
                workers=WARC_WORKERS
            ),
            # Filter truncated PDFs
            PDFTruncationDetector(),
            # Classify and route PDFs
            PDFRouter(
                model_path=MODEL_PATH,
                threshold=OCR_THRESHOLD
            ),
            # Save ALL PDFs with routing metadata
            JsonlWriter(CLASSIFIED_OUTPUT, save_media_bytes=True),
        ],
        tasks=WARC_WORKERS,
        logging_dir=f"{LOGGING_DIR}/classification"
    )

    # ========================================================================
    # Stage 2: Text Extraction Path - Low OCR probability PDFs
    # ========================================================================
    print("\nStage 2: Text Extraction (Low OCR Probability)")
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
        tasks=TEXT_EXTRACTION_TASKS,
        logging_dir=f"{LOGGING_DIR}/text_extraction",
        depends=stage1_classification  # Wait for classification to complete
    )

    # ========================================================================
    # Stage 3: OCR Extraction Path - High OCR probability PDFs
    # ========================================================================
    print("\nStage 3: OCR Extraction (High OCR Probability)")
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
                    PostProcessOCRResults(),
                    PersistentContextJsonlWriter(OCR_EXTRACTION_OUTPUT)
                ]
            ),
        ],
        tasks=OCR_EXTRACTION_TASKS,  # GPU-bound, single task
        logging_dir=f"{LOGGING_DIR}/ocr_extraction",
        depends=stage1_classification  # Wait for classification to complete
    )

    return stage1_classification, stage2_text_extraction, stage3_ocr_extraction


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run the complete FinePDFs production pipeline."""

    # Check prerequisites
    if not Path(WARC_METADATA_FILE).exists():
        print(f"❌ Error: WARC metadata file not found: {WARC_METADATA_FILE}")
        print("\nCreate this file with JSONL entries containing:")
        print('  {"id": "...", "warc_filename": "...", "warc_record_offset": ...}')
        return

    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: XGBoost model not found: {MODEL_PATH}")
        print("\nTrain the model using examples_local/08b_pdf_classifier_model.py")
        return

    # Create pipeline
    stage1, stage2, stage3 = create_production_pipeline()

    # Run stages (2 & 3 will run in parallel after stage 1 completes)
    print("\n" + "=" * 80)
    print("Starting Pipeline Execution")
    print("=" * 80)
    print("\nStage 1 will run first (classification)")
    print("Stages 2 & 3 will run in parallel after Stage 1 completes\n")

    try:
        # Run stage 3 (which depends on stage 1)
        # Stage 2 and 3 will automatically wait for stage 1
        stage3.run()
    finally:
        # Explicitly close the writer to ensure gzip file is properly finalized
        writer = None
        for step in stage3.pipeline:
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


if __name__ == "__main__":
    main()
