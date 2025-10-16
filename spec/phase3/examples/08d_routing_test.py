#!/usr/bin/env python3
"""
Example 08d: PDF Routing Test

Tests PDF routing logic with sample PDFs.

Components:
- PDFRouter: Route PDFs based on OCR probability
- LambdaFilter: Filter by routing decision
- JsonlWriter: Save routed PDFs

Usage:
    python spec/phase3/examples/08d_routing_test.py
"""

from pathlib import Path

from datatrove.data import Document, Media, MediaType
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from datatrove.pipeline.filters.pdf_router import PDFRouter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.logging import logger

# Sample PDFs to test routing - use threshold analysis samples
SAMPLE_PDFS = [
    # Low OCR probability samples (should route to text_extraction)
    "spec/phase3/threshold_analysis/samples/low_ocr/low_ocr_01_<urn:uuid:449f2fe2-49b5-4609-a4c9-901ebbffbb81>.pdf",
    "spec/phase3/threshold_analysis/samples/low_ocr/low_ocr_02_<urn:uuid:12fcdb36-1e9d-4192-88c8-55a70ec2872f>.pdf",
    "spec/phase3/threshold_analysis/samples/low_ocr/low_ocr_03_<urn:uuid:ead811e4-4126-4ef9-8525-38beb86665a4>.pdf",
    # High OCR probability samples (should route to ocr_extraction)
    "spec/phase3/threshold_analysis/samples/high_ocr/high_ocr_01_<urn:uuid:98e53922-1ff8-45fd-be5c-41d9f906e869>.pdf",
    "spec/phase3/threshold_analysis/samples/high_ocr/high_ocr_02_<urn:uuid:f808a467-bd86-4c90-9e50-eeb5d47d36b5>.pdf",
    "spec/phase3/threshold_analysis/samples/high_ocr/high_ocr_03_<urn:uuid:3c02344a-24d1-4e38-961f-8b1f7bee9e32>.pdf",
]

# Configuration
MODEL_PATH = "spec/phase3/data/pdf_classifier_real_data.xgb"
OUTPUT_DIR = "spec/phase3/output/routing_test"


def load_pdf_documents():
    """Load sample PDFs as Document objects."""
    documents = []
    for pdf_path in SAMPLE_PDFS:
        path = Path(pdf_path)
        if not path.exists():
            logger.info(f"Warning: {pdf_path} not found, skipping")
            continue

        with open(path, "rb") as f:
            pdf_bytes = f.read()

        doc = Document(
            text="",  # Empty until extracted
            id=path.stem,
            media=[
                Media(
                    id=path.stem,
                    type=MediaType.DOCUMENT,
                    media_bytes=pdf_bytes,  # Correct: PDF bytes in Media object
                    url=f"file://{path}",
                )
            ],
            metadata={"source": str(path)}
        )
        documents.append(doc)

    return documents


def test_routing():
    """Test PDF routing with XGBoost classifier."""

    logger.info("Stage 1: Classification - Route PDFs based on OCR probability")

    # Load PDFs
    documents = load_pdf_documents()
    logger.info(f"Loaded {len(documents)} PDFs for testing")

    # Stage 1: Classify and route PDFs
    stage1_classification = LocalPipelineExecutor(
        pipeline=[
            documents,  # Start with our document list
            PDFRouter(
                model_path=MODEL_PATH,
                threshold=0.5
            ),
            JsonlWriter(OUTPUT_DIR + "/classified", save_media_bytes=True),  # Save Media objects with PDF bytes
        ],
        tasks=1,
        logging_dir="spec/phase3/logs/routing_test/classification"
    )

    stage1_classification.run()

    logger.info("Stage 2: Text Extraction Path (Low OCR Probability)")

    # Stage 2: Filter and process low OCR PDFs
    from datatrove.pipeline.readers import JsonlReader

    stage2_text_extraction = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(OUTPUT_DIR + "/classified"),
            LambdaFilter(
                filter_function=lambda doc: doc.metadata.get("processing_route") == "text_extraction"
            ),
            JsonlWriter(OUTPUT_DIR + "/text_extraction"),
        ],
        tasks=1,
        logging_dir="spec/phase3/logs/routing_test/text_extraction",
        depends=stage1_classification
    )

    stage2_text_extraction.run()

    logger.info("Stage 3: OCR Extraction Path (High OCR Probability)")

    # Stage 3: Filter and process high OCR PDFs
    stage3_ocr_extraction = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(OUTPUT_DIR + "/classified"),
            LambdaFilter(
                filter_function=lambda doc: doc.metadata.get("processing_route") == "ocr_extraction"
            ),
            JsonlWriter(OUTPUT_DIR + "/ocr_extraction"),
        ],
        tasks=1,
        logging_dir="spec/phase3/logs/routing_test/ocr_extraction",
        depends=stage1_classification
    )

    stage3_ocr_extraction.run()

    logger.info("Routing Test Complete!")
    logger.info(f"Classified PDFs: {OUTPUT_DIR}/classified")
    logger.info(f"Text Extraction Path: {OUTPUT_DIR}/text_extraction")
    logger.info(f"OCR Extraction Path: {OUTPUT_DIR}/ocr_extraction")
    logger.info("Check logs for routing statistics: spec/phase3/logs/routing_test/classification/stats/")


if __name__ == "__main__":
    test_routing()
