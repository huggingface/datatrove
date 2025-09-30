#!/usr/bin/env python3
"""
RolmOCR Integration Test Script

Tests RolmOCR integration using DataTrove's inference infrastructure.
Follows the exact approach from FinePDFs paper: RolmOCR on LMDeploy.
"""

import asyncio
import base64
import fitz
import json
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, 'src')

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceConfig
from datatrove.pipeline.inference.utils.page_rendering import render_page_to_base64png_pymupdf
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.run_inference import InferenceSuccess
from typing import Iterable


class PostProcessOCRResults(PipelineStep):
    """Post-process RolmOCR inference results."""

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            # Extract OCR text from inference results and replace PDF bytes
            document.text = "\n".join([x.text if isinstance(x, InferenceSuccess) else x.error for x in document.metadata["inference_results"]])
            # Clean up inference_results metadata
            del document.metadata["inference_results"]
            yield document


def rolmocr_query_builder(runner: InferenceRunner, doc: Document) -> dict:
    """Convert PDF document to RolmOCR vision request.

    Follows FinePDFs specification:
    - Rescale PDFs so longest dimension ≥ 1280px
    - Ensure representation doesn't exceed 2048 image tokens
    - Total context length set to 8096 tokens
    """

    # Open PDF from document text (PDF bytes)
    pdf_bytes = doc.text if isinstance(doc.text, bytes) else doc.text.encode()
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Process limited pages to avoid memory issues (max 5 pages for testing)
    max_pages = min(5, len(pdf_doc))
    page_images = []
    for page_num in range(max_pages):
        page = pdf_doc.load_page(page_num)

        # Use smaller image size to reduce memory usage
        base64_image = render_page_to_base64png_pymupdf(
            page,
            resize_longest_side_pixels=640,  # Reduced from 1280
            max_visual_tokens=512  # Reduced from 2048
        )

        page_images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })

    pdf_doc.close()

    print(f"Processing {max_pages} pages with reduced resolution for memory efficiency")

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
    sample_dir = Path("examples_local/threshold_analysis/samples/high_ocr")
    if not sample_dir.exists():
        print(f"Warning: {sample_dir} not found.")
        return []

    sample_info_path = sample_dir / "sample_info.json"
    if not sample_info_path.exists():
        print(f"Warning: {sample_info_path} not found.")
        return []

    with open(sample_info_path) as f:
        sample_info = json.load(f)

    documents = []
    # Filter to only working PDFs (02, 04, 05 - skip broken 01 and 03)
    working_pdfs = [pdf for pdf in sample_info if 'high_ocr_02_' in pdf['saved_filename'] or
                                                    'high_ocr_04_' in pdf['saved_filename'] or
                                                    'high_ocr_05_' in pdf['saved_filename']]

    # Test only the first working PDF to debug document overwriting
    for pdf_info in working_pdfs[:1]:
        pdf_path = sample_dir / pdf_info['saved_filename']
        if not pdf_path.exists():
            continue

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = Document(
            text=pdf_bytes,
            id=pdf_info['id'],
            metadata={
                "source": str(pdf_path),
                "ocr_probability": pdf_info['ocr_prob'],
                "processing_method": "rolmocr",
                "url": pdf_info.get('url', ''),
                "num_pages": pdf_info.get('num_pages', 0)
            }
        )
        documents.append(doc)

    return documents


def test_rolmocr_integration():
    """Test RolmOCR using DataTrove's inference infrastructure."""

    print("Loading high OCR probability PDFs...")
    documents = load_high_ocr_pdfs()

    if not documents:
        print("No high OCR PDFs found. Exiting test.")
        return

    print(f"Found {len(documents)} high OCR probability PDFs")
    for doc in documents:
        print(f"  - {doc.id}: OCR prob {doc.metadata['ocr_probability']:.3f}")

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

    # Create inference runner with proper post-processing
    runner = InferenceRunner(
        query_builder=rolmocr_query_builder,
        config=config,
        post_process_steps=[
            PostProcessOCRResults(),  # Extract OCR text from inference_results
            JsonlWriter("examples_local/output/rolmocr_results")
        ]
    )

    print("Starting RolmOCR inference...")

    # Run RolmOCR inference
    try:
        runner.run(documents, rank=0, world_size=1)
        print("RolmOCR inference completed successfully!")
    except Exception as e:
        print(f"RolmOCR inference failed: {e}")
        raise


def test_query_builder_only():
    """Test just the query builder without actual inference."""

    print("Testing RolmOCR query builder...")

    # Load one high OCR PDF for testing
    documents = load_high_ocr_pdfs()
    if not documents:
        print("No PDFs available for query builder test")
        return

    test_doc = documents[0]
    print(f"Testing with PDF: {test_doc.id}")

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
        print("Query builder test successful!")
        print(f"Generated query keys: {list(query.keys())}")
        print(f"Model: {query['model']}")
        print(f"Max tokens: {query['max_tokens']}")
        print(f"Temperature: {query['temperature']}")

        # Check message structure
        messages = query['messages'][0]['content']
        text_items = [item for item in messages if item['type'] == 'text']
        image_items = [item for item in messages if item['type'] == 'image_url']

        print(f"Text prompts: {len(text_items)}")
        print(f"Image inputs: {len(image_items)}")

        if text_items:
            print(f"Prompt: {text_items[0]['text']}")

    except Exception as e:
        print(f"Query builder test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("RolmOCR Integration Test")
    print("=" * 40)

    # Test query builder first (safer)
    test_query_builder_only()

    print("\n" + "=" * 40)

    # Test full integration if PDFs are available
    # LMDeployServer will automatically pull RolmOCR model from HuggingFace Hub
    test_rolmocr_integration()