#!/usr/bin/env python3
"""
Example 08c: RolmOCR Integration Test

Tests RolmOCR integration using DataTrove's inference infrastructure.

Components:
- InferenceRunner: Run OCR inference on PDFs
- rolmocr_query_builder: Convert PDFs to RolmOCR vision requests
- PersistentContextJsonlWriter: Save results

Usage:
    python spec/phase3/examples/08c_rolmocr_test.py
"""

import asyncio
import base64
import json
from pathlib import Path
from typing import Iterable, List

import fitz

from datatrove.data import Document, Media, MediaType
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.post_process import ExtractInferenceText
from datatrove.pipeline.inference.query_builders.vision import rolmocr_query_builder
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.inference.utils.page_rendering import render_page_to_base64png_pymupdf
from datatrove.pipeline.writers.jsonl import PersistentContextJsonlWriter
from datatrove.utils.logging import logger

# Configuration
OUTPUT_DIR = "spec/phase3/output/rolmocr_test"
LOGS_DIR = "spec/phase3/logs/rolmocr_test"


def rolmocr_query_builder_with_debug(runner: InferenceRunner, doc: Document) -> dict:
    """Convert PDF document to RolmOCR vision request.

    Follows FinePDFs specification:
    - Rescale PDFs so longest dimension ≥ 1280px
    - Ensure representation doesn't exceed 2048 image tokens
    - Total context length set to 8096 tokens
    """

    # Get PDF bytes from Media object
    if not doc.media or not doc.media[0].media_bytes:
        raise ValueError(f"Document {doc.id} has no media bytes")

    pdf_bytes = doc.media[0].media_bytes
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Process limited pages to avoid memory issues (max 5 pages for testing)
    max_pages = min(5, len(pdf_doc))
    page_images = []

    # Create debug directory for saving processed images
    debug_dir = Path("spec/phase3/output/rolmocr_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    for page_num in range(max_pages):
        page = pdf_doc.load_page(page_num)

        # Use FinePDFs specification resolution
        base64_image = render_page_to_base64png_pymupdf(
            page,
            resize_longest_side_pixels=1280,  # FinePDFs spec
            max_visual_tokens=2048  # FinePDFs spec
        )

        # Save processed image to file for debugging
        doc_id_safe = doc.id.replace('/', '_').replace(':', '_')
        image_path = debug_dir / f"{doc_id_safe}_page_{page_num}.png"
        with open(image_path, 'wb') as f:
            f.write(base64.b64decode(base64_image))
        logger.info(f"Saved processed image: {image_path}")

        page_images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })

    pdf_doc.close()

    logger.info(f"Processing {max_pages} pages with reduced resolution for memory efficiency")

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


def load_high_ocr_pdfs() -> List[Document]:
    """Load high OCR probability PDFs for testing."""

    # Look for high OCR probability PDFs from previous classification
    sample_dir = Path("spec/phase3/threshold_analysis/samples/high_ocr")
    if not sample_dir.exists():
        logger.info(f"Warning: {sample_dir} not found.")
        return []

    sample_info_path = sample_dir / "sample_info.json"
    if not sample_info_path.exists():
        logger.info(f"Warning: {sample_info_path} not found.")
        return []

    with open(sample_info_path) as f:
        sample_info = json.load(f)

    documents = []
    # Filter to only working PDFs (02, 04, 05 - skip broken 01 and 03)
    working_pdfs = [pdf for pdf in sample_info if 'high_ocr_02_' in pdf['saved_filename'] or
                                                    'high_ocr_04_' in pdf['saved_filename'] or
                                                    'high_ocr_05_' in pdf['saved_filename']]

    # Test all working PDFs
    for pdf_info in working_pdfs:
        pdf_path = sample_dir / pdf_info['saved_filename']
        if not pdf_path.exists():
            continue

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = Document(
            text="",  # Empty until OCR extraction
            id=pdf_info['id'],
            media=[
                Media(
                    id=pdf_info['id'],
                    type=MediaType.DOCUMENT,
                    media_bytes=pdf_bytes,  # Correct: PDF bytes in Media object
                    url=pdf_info.get('url', f"file://{pdf_path}"),
                )
            ],
            metadata={
                "source": str(pdf_path),
                "ocr_probability": pdf_info['ocr_prob'],
                "processing_method": "rolmocr",
                "num_pages": pdf_info.get('num_pages', 0)
            }
        )
        documents.append(doc)

    return documents


def test_rolmocr_integration():
    """Test RolmOCR using DataTrove's inference infrastructure."""

    logger.info("Loading high OCR probability PDFs...")
    documents = load_high_ocr_pdfs()

    if not documents:
        logger.info("No high OCR PDFs found. Exiting test.")
        return

    logger.info(f"Found {len(documents)} high OCR probability PDFs")
    for doc in documents:
        logger.info(f"  - {doc.id}: OCR prob {doc.metadata['ocr_probability']:.3f}")

    # Configure RolmOCR with LMDeploy
    config = InferenceConfig(
        server_type="lmdeploy",
        model_name_or_path="Reducto/RolmOCR",  # Or actual model path
        model_max_context=8096,  # From FinePDFs paper
        max_concurrent_requests=1,
        max_concurrent_tasks=1,
        model_kwargs={
            "chat_template": "internlm",
            "vision_max_batch_size": 128  # From LMDeployServer
        }
    )

    # Create pipeline executor (following the inference_example_basic.py pattern)
    pipeline_executor = LocalPipelineExecutor(
        pipeline=[
            documents,  # Documents as part of the pipeline
            InferenceRunner(
                query_builder=rolmocr_query_builder_with_debug,
                config=config,
                post_process_steps=[
                    ExtractInferenceText(),  # Extract OCR text from inference_results
                    PersistentContextJsonlWriter("spec/phase3/output/rolmocr_results")
                ]
            ),
        ],
        logging_dir=None,
    )

    logger.info("Starting RolmOCR inference...")

    # Run the pipeline executor
    try:
        pipeline_executor.run()
        logger.info("RolmOCR inference completed successfully!")
    except Exception as e:
        logger.info(f"RolmOCR inference failed: {e}")
        raise
    finally:
        # Explicitly close the writer to ensure gzip file is properly finalized
        writer = None
        for step in pipeline_executor.pipeline:
            if isinstance(step, InferenceRunner):
                for post_step in step.post_process_steps:
                    if isinstance(post_step, PersistentContextJsonlWriter):
                        writer = post_step
                        break
        if writer and writer._context_entered:
            logger.info("Closing writer context...")
            writer.__exit__(None, None, None)


def test_query_builder_only():
    """Test just the query builder without actual inference."""

    logger.info("Testing RolmOCR query builder...")

    # Load one high OCR PDF for testing
    documents = load_high_ocr_pdfs()
    if not documents:
        logger.info("No PDFs available for query builder test")
        return

    test_doc = documents[0]
    logger.info(f"Testing with PDF: {test_doc.id}")

    # Mock runner for testing
    class MockRunner:
        def __init__(self):
            self.config = InferenceConfig(
                server_type="lmdeploy",
                model_name_or_path="Reducto/RolmOCR",
                model_max_context=8096,
                max_concurrent_requests=1,
                max_concurrent_tasks=1
            )

    mock_runner = MockRunner()

    try:
        query = rolmocr_query_builder(mock_runner, test_doc)
        logger.info("Query builder test successful!")
        logger.info(f"Generated query keys: {list(query.keys())}")
        logger.info(f"Model: {query['model']}")
        logger.info(f"Max tokens: {query['max_tokens']}")
        logger.info(f"Temperature: {query['temperature']}")

        # Check message structure
        messages = query['messages'][0]['content']
        text_items = [item for item in messages if item['type'] == 'text']
        image_items = [item for item in messages if item['type'] == 'image_url']

        logger.info(f"Text prompts: {len(text_items)}")
        logger.info(f"Image inputs: {len(image_items)}")

        if text_items:
            logger.info(f"Prompt: {text_items[0]['text']}")

    except Exception as e:
        logger.info(f"Query builder test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logger.info("RolmOCR Integration Test")

    # Test query builder first (safer)
    test_query_builder_only()


    # Test full integration if PDFs are available
    # LMDeployServer will automatically pull RolmOCR model from HuggingFace Hub
    test_rolmocr_integration()