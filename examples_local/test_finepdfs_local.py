#!/usr/bin/env python3
"""
Test FinePDFs Pipeline with Local PDFs

Uses the same local PDFs as test_routing.py but adds full extraction:
- Stage 1: Classification with PDFRouter
- Stage 2: Docling extraction for low OCR PDFs
- Stage 3: RolmOCR extraction for high OCR PDFs
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from datatrove.data import Document, Media, MediaType
from datatrove.pipeline.filters.pdf_router import PDFRouter
from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from datatrove.pipeline.media.extractors.extractors import DoclingExtractor
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceConfig, InferenceSuccess
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.readers import JsonlReader
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from typing import Iterable
import fitz
from datatrove.pipeline.inference.utils.page_rendering import render_page_to_base64png_pymupdf


# Sample PDFs - same as test_routing.py
SAMPLE_PDFS = [
    # Low OCR probability samples (should route to text_extraction)
    "examples_local/threshold_analysis/samples/low_ocr/low_ocr_01_<urn:uuid:449f2fe2-49b5-4609-a4c9-901ebbffbb81>.pdf",
    "examples_local/threshold_analysis/samples/low_ocr/low_ocr_02_<urn:uuid:12fcdb36-1e9d-4192-88c8-55a70ec2872f>.pdf",
    "examples_local/threshold_analysis/samples/low_ocr/low_ocr_03_<urn:uuid:ead811e4-4126-4ef9-8525-38beb86665a4>.pdf",
    # High OCR probability samples (should route to ocr_extraction)
    "examples_local/threshold_analysis/samples/high_ocr/high_ocr_02_<urn:uuid:f808a467-bd86-4c90-9e50-eeb5d47d36b5>.pdf",
    "examples_local/threshold_analysis/samples/high_ocr/high_ocr_04_<urn:uuid:24d2dd9d-271d-49fb-8817-abc5f42e46c0>.pdf",
    "examples_local/threshold_analysis/samples/high_ocr/high_ocr_05_<urn:uuid:38bc9a73-50e9-4744-af6a-0f357ed5721c>.pdf",
]

# Configuration
MODEL_PATH = "examples_local/pdf_classifier_real_data.xgb"
CLASSIFIED_OUTPUT = "examples_local/output/finepdfs_local/classified"
TEXT_EXTRACTION_OUTPUT = "examples_local/output/finepdfs_local/text_extraction"
OCR_EXTRACTION_OUTPUT = "examples_local/output/finepdfs_local/ocr_extraction"


# ============================================================================
# RolmOCR Support Classes
# ============================================================================

class PostProcessOCRResults(PipelineStep):
    """Post-process RolmOCR inference results."""

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            # Extract OCR text from inference results
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


class SavePDFsToDisk(PipelineStep):
    """Save PDF bytes from Media objects to disk as .pdf files."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            # Save PDF to disk if media bytes exist
            if document.media and document.media[0].media_bytes:
                pdf_path = self.output_dir / f"{document.id}.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(document.media[0].media_bytes)
                self.stat_update("pdfs_saved")
            yield document


class SaveOCRPagesAsPNG(PipelineStep):
    """Save rendered PDF pages as PNG images (as sent to RolmOCR)."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        import base64
        for document in data:
            if document.media and document.media[0].media_bytes:
                pdf_bytes = document.media[0].media_bytes
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

                # Render pages (match RolmOCR processing)
                max_pages = min(1, len(pdf_doc))  # Same as rolmocr_query_builder
                for page_num in range(max_pages):
                    page = pdf_doc.load_page(page_num)

                    # Use same rendering as RolmOCR
                    base64_image = render_page_to_base64png_pymupdf(
                        page,
                        resize_longest_side_pixels=1280,
                        max_visual_tokens=2048
                    )

                    # Save PNG
                    png_path = self.output_dir / f"{document.id}_page{page_num + 1:03d}.png"
                    with open(png_path, "wb") as f:
                        f.write(base64.b64decode(base64_image))

                    self.stat_update("pages_saved")

                pdf_doc.close()
            yield document


def rolmocr_query_builder(runner: InferenceRunner, doc: Document) -> dict:
    """Convert PDF document to RolmOCR vision request."""

    # Get PDF bytes from Media object
    if not doc.media or not doc.media[0].media_bytes:
        raise ValueError(f"Document {doc.id} has no media bytes")

    pdf_bytes = doc.media[0].media_bytes
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Process limited pages for testing (first page only)
    max_pages = min(1, len(pdf_doc))
    page_images = []

    for page_num in range(max_pages):
        page = pdf_doc.load_page(page_num)

        # Use FinePDFs specification resolution
        base64_image = render_page_to_base64png_pymupdf(
            page,
            resize_longest_side_pixels=1280,
            max_visual_tokens=2048
        )

        page_images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })

    pdf_doc.close()

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
        "max_tokens": 4096,
        "temperature": 0.0
    }


# ============================================================================
# Load Local PDFs
# ============================================================================

def load_pdf_documents():
    """Load sample PDFs as Document objects with Media."""
    documents = []
    for pdf_path in SAMPLE_PDFS:
        path = Path(pdf_path)
        if not path.exists():
            print(f"Warning: {pdf_path} not found, skipping")
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
                    media_bytes=pdf_bytes,
                    url=f"file://{path}",
                )
            ],
            metadata={"source": str(path)}
        )
        documents.append(doc)

    return documents


# ============================================================================
# Pipeline
# ============================================================================

def test_finepdfs_local():
    """Test FinePDFs pipeline with local PDFs."""

    print("=" * 80)
    print("FinePDFs Local Test - Two-Tiered PDF Processing")
    print("=" * 80)

    # Load PDFs
    documents = load_pdf_documents()
    print(f"\nLoaded {len(documents)} PDFs for testing")

    # ========================================================================
    # Stage 1: Classification
    # ========================================================================
    print("\n" + "=" * 80)
    print("Stage 1: PDF Classification and Routing")
    print("=" * 80)

    stage1_classification = LocalPipelineExecutor(
        pipeline=[
            documents,
            PDFRouter(
                model_path=MODEL_PATH,
                threshold=0.5
            ),
            JsonlWriter(CLASSIFIED_OUTPUT, save_media_bytes=True),
        ],
        tasks=1,
        logging_dir="examples_local/logs/finepdfs_local/classification"
    )

    stage1_classification.run()

    # ========================================================================
    # Stage 2: Text Extraction (Low OCR)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Stage 2: Text Extraction (Low OCR Probability)")
    print("=" * 80)

    stage2_text_extraction = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(CLASSIFIED_OUTPUT),
            LambdaFilter(
                filter_function=lambda doc: doc.metadata.get("processing_route") == "text_extraction"
            ),
            SavePDFsToDisk("examples_local/output/finepdfs_local/text_extraction_pdfs"),
            DoclingExtractor(timeout=60),
            JsonlWriter(TEXT_EXTRACTION_OUTPUT),
        ],
        tasks=1,
        logging_dir="examples_local/logs/finepdfs_local/text_extraction",
        depends=stage1_classification
    )

    stage2_text_extraction.run()

    # ========================================================================
    # Stage 3: OCR Extraction (High OCR)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Stage 3: OCR Extraction (High OCR Probability)")
    print("=" * 80)

    stage3_ocr_extraction = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(CLASSIFIED_OUTPUT),
            LambdaFilter(
                filter_function=lambda doc: doc.metadata.get("processing_route") == "ocr_extraction"
            ),
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
                    SavePDFsToDisk("examples_local/output/finepdfs_local/ocr_extraction_pdfs"),
                    SaveOCRPagesAsPNG("examples_local/output/finepdfs_local/ocr_extraction_pages_png"),
                    PersistentContextJsonlWriter(OCR_EXTRACTION_OUTPUT)
                ]
            ),
        ],
        tasks=1,
        logging_dir="examples_local/logs/finepdfs_local/ocr_extraction",
        depends=stage1_classification
    )

    stage3_ocr_extraction.run()

    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  Classified: {CLASSIFIED_OUTPUT}")
    print(f"  Text Extraction: {TEXT_EXTRACTION_OUTPUT}")
    print(f"  OCR Extraction: {OCR_EXTRACTION_OUTPUT}")


if __name__ == "__main__":
    test_finepdfs_local()
