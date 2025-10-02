"""Query builders for vision models (OCR, image understanding, etc.)."""

import fitz

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import InferenceRunner


def rolmocr_query_builder(runner: InferenceRunner, doc: Document) -> dict:
    """Convert PDF document to RolmOCR vision request.

    Follows FinePDFs specification:
    - Rescale PDFs so longest dimension ≥ 1280px
    - Ensure representation doesn't exceed 2048 image tokens
    - Total context length set to 8096 tokens

    Args:
        runner: InferenceRunner instance (provides config.model_name_or_path)
        doc: Document with Media object containing PDF bytes

    Returns:
        OpenAI-compatible vision request dict with base64-encoded page images

    Raises:
        ValueError: If document has no media bytes
    """
    from datatrove.pipeline.inference.utils.page_rendering import render_page_to_base64png_pymupdf

    # Get PDF bytes from Media object
    if not doc.media or not doc.media[0].media_bytes:
        raise ValueError(f"Document {doc.id} has no media bytes")

    pdf_bytes = doc.media[0].media_bytes
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Process all pages (or limit for memory/cost)
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
