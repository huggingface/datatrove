#!/usr/bin/env python3
"""
Example 08: FinePDFs Pipeline with Local WARC Files

Complete end-to-end PDF processing pipeline using local WARC files.

Components:
- PDFWarcReader: Extract PDFs from WARC files
- PDFTruncationFilter: Filter truncated PDFs
- PDFRouter: Classify PDFs by OCR probability
- DoclingExtractor: Extract text from low OCR PDFs
- RolmOCR: Extract text from high OCR PDFs via inference

Usage:
    python spec/phase3/examples/08_finepdfs_warc.py
"""

from pathlib import Path

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from datatrove.pipeline.filters.pdf_router import PDFRouter
from datatrove.pipeline.filters.pdf_truncation import PDFTruncationFilter
from datatrove.pipeline.inference.post_process import ExtractInferenceText
from datatrove.pipeline.inference.query_builders.vision import rolmocr_query_builder
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.media.extractors.extractors import DoclingExtractor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.readers.pdf_warc import PDFWarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter, PersistentContextJsonlWriter
from datatrove.utils.logging import logger

# Configuration
WARC_DATA_FOLDER = "spec/phase3/data"
MODEL_PATH = "spec/phase3/data/pdf_classifier_real_data.xgb"
OUTPUT_DIR = "spec/phase3/output/finepdfs_warc"
LOGS_DIR = "spec/phase3/logs/finepdfs_warc"

OCR_THRESHOLD = 0.5


def main():
    """Test full FinePDFs pipeline with local WARC files."""

    logger.info("FinePDFs Pipeline Test - WARC Files")
    logger.info(f"Configuration: WARC Files={WARC_DATA_FOLDER}/*.warc.gz, Model={MODEL_PATH}, OCR Threshold={OCR_THRESHOLD}, Output={OUTPUT_DIR}")

    # ========================================================================
    # Stage 1: Classification - Extract PDFs from WARCs and route
    # ========================================================================
    logger.info("Stage 1: PDF Extraction, Truncation Detection, and Classification")

    stage1_classification = LocalPipelineExecutor(
        pipeline=[
            # Extract PDFs from WARC files
            PDFWarcReader(
                data_folder=WARC_DATA_FOLDER,
                glob_pattern="*.warc.gz",
                limit=100  # Limit for testing
            ),
            # Filter truncated PDFs
            PDFTruncationFilter(),
            # Classify and route PDFs
            PDFRouter(
                model_path=MODEL_PATH,
                threshold=OCR_THRESHOLD
            ),
            # Save ALL PDFs with routing metadata
            JsonlWriter(OUTPUT_DIR + "/classified", save_media_bytes=True),
        ],
        tasks=1,  # Single task for local testing
        logging_dir=LOGS_DIR + "/classification"
    )

    # ========================================================================
    # Execute Pipeline
    # ========================================================================
    logger.info("Starting Pipeline Execution")
    logger.info("Running Stage 1: Classification")

    try:
        # Run stage 1 first
        stage1_classification.run()

        # ========================================================================
        # Stage 2: Text Extraction Path - Low OCR probability PDFs
        # ========================================================================
        logger.info("Stage 2: Text Extraction (Low OCR Probability - Docling)")

        stage2_text_extraction = LocalPipelineExecutor(
            pipeline=[
                # Read pre-classified PDFs with Media objects
                JsonlReader(OUTPUT_DIR + "/classified", glob_pattern="*.jsonl.gz"),
                # Filter for low OCR PDFs
                LambdaFilter(
                    filter_function=lambda doc: doc.metadata.get("processing_route") == "text_extraction"
                ),
                # Extract text with Docling
                DoclingExtractor(timeout=10*60),  # 10 minute timeout per PDF
                # Save extracted text
                JsonlWriter(OUTPUT_DIR + "/text_extraction"),
            ],
            tasks=1,
            logging_dir=LOGS_DIR + "/text_extraction"
        )

        # ========================================================================
        # Stage 3: OCR Extraction Path - High OCR probability PDFs
        # ========================================================================
        logger.info("Stage 3: OCR Extraction (High OCR Probability - RolmOCR)")

        stage3_ocr_extraction = LocalPipelineExecutor(
            pipeline=[
                # Read same pre-classified PDFs
                JsonlReader(OUTPUT_DIR + "/classified", glob_pattern="*.jsonl.gz"),
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
                        PersistentContextJsonlWriter(OUTPUT_DIR + "/ocr_extraction")
                    ]
                ),
            ],
            tasks=1,  # GPU-bound, single task
            logging_dir=LOGS_DIR + "/ocr_extraction"
        )

        # Run stages 2 & 3
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
            logger.info("Closing OCR writer context...")
            writer.__exit__(None, None, None)

    logger.info("Pipeline Completed Successfully!")
    logger.info(f"Outputs:")
    logger.info(f"  Classified PDFs: {OUTPUT_DIR}/classified")
    logger.info(f"  Text Extraction: {OUTPUT_DIR}/text_extraction")
    logger.info(f"  OCR Extraction: {OUTPUT_DIR}/ocr_extraction")
    logger.info(f"Logs: {LOGS_DIR}/*/stats/")
    logger.info("Run extract_text_for_review.py to extract text from results")


if __name__ == "__main__":
    main()
