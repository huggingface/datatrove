"""Test PDF routing logic with sample PDFs."""

from pathlib import Path
from datatrove.data import Document
from datatrove.pipeline.filters.pdf_router import PDFRouter
from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.executor.local import LocalPipelineExecutor

# Sample PDFs to test routing
SAMPLE_PDFS = [
    "examples_local/data/pdfs/CC-MAIN-20240916-02.pdf",  # OCR prob: 0.001 (very low)
    "examples_local/data/pdfs/CC-MAIN-20240916-04.pdf",  # OCR prob: 0.003 (low)
    "examples_local/data/pdfs/CC-MAIN-20240916-05.pdf",  # OCR prob: 0.758 (high)
]

# XGBoost model path
MODEL_PATH = "examples_local/pdf_classifier_real_data.xgb"

# Output paths
CLASSIFIED_OUTPUT = "examples_local/output/routing_test/classified"
TEXT_EXTRACTION_OUTPUT = "examples_local/output/routing_test/text_extraction"
OCR_EXTRACTION_OUTPUT = "examples_local/output/routing_test/ocr_extraction"


def load_pdf_documents():
    """Load sample PDFs as Document objects."""
    documents = []
    for pdf_path in SAMPLE_PDFS:
        path = Path(pdf_path)
        if not path.exists():
            print(f"Warning: {pdf_path} not found, skipping")
            continue

        with open(path, "rb") as f:
            pdf_bytes = f.read()

        doc = Document(
            text=pdf_bytes,
            id=path.stem,
            metadata={"source": str(path)}
        )
        documents.append(doc)

    return documents


def test_routing():
    """Test PDF routing with XGBoost classifier."""

    print("=" * 60)
    print("Stage 1: Classification - Route PDFs based on OCR probability")
    print("=" * 60)

    # Load PDFs
    documents = load_pdf_documents()
    print(f"\nLoaded {len(documents)} PDFs for testing")

    # Stage 1: Classify and route PDFs
    stage1_classification = LocalPipelineExecutor(
        pipeline=[
            documents,  # Start with our document list
            PDFRouter(
                model_path=MODEL_PATH,
                threshold=0.5
            ),
            JsonlWriter(CLASSIFIED_OUTPUT),
        ],
        tasks=1,
        logging_dir="examples_local/logs/routing_test/classification"
    )

    stage1_classification.run()

    print("\n" + "=" * 60)
    print("Stage 2: Text Extraction Path (Low OCR Probability)")
    print("=" * 60)

    # Stage 2: Filter and process low OCR PDFs
    from datatrove.pipeline.readers import JsonlReader

    stage2_text_extraction = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(CLASSIFIED_OUTPUT),
            LambdaFilter(
                filter_function=lambda doc: doc.metadata.get("processing_route") == "text_extraction"
            ),
            JsonlWriter(TEXT_EXTRACTION_OUTPUT),
        ],
        tasks=1,
        logging_dir="examples_local/logs/routing_test/text_extraction",
        depends=stage1_classification
    )

    stage2_text_extraction.run()

    print("\n" + "=" * 60)
    print("Stage 3: OCR Extraction Path (High OCR Probability)")
    print("=" * 60)

    # Stage 3: Filter and process high OCR PDFs
    stage3_ocr_extraction = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(CLASSIFIED_OUTPUT),
            LambdaFilter(
                filter_function=lambda doc: doc.metadata.get("processing_route") == "ocr_extraction"
            ),
            JsonlWriter(OCR_EXTRACTION_OUTPUT),
        ],
        tasks=1,
        logging_dir="examples_local/logs/routing_test/ocr_extraction",
        depends=stage1_classification
    )

    stage3_ocr_extraction.run()

    print("\n" + "=" * 60)
    print("Routing Test Complete!")
    print("=" * 60)
    print(f"\nClassified PDFs: {CLASSIFIED_OUTPUT}")
    print(f"Text Extraction Path: {TEXT_EXTRACTION_OUTPUT}")
    print(f"OCR Extraction Path: {OCR_EXTRACTION_OUTPUT}")
    print("\nCheck logs for routing statistics:")
    print("  - examples_local/logs/routing_test/classification/stats/")


if __name__ == "__main__":
    test_routing()
