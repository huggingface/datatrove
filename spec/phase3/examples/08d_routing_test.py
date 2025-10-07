"""Test PDF routing logic with sample PDFs."""

from pathlib import Path
from datatrove.data import Document, Media, MediaType
from datatrove.pipeline.filters.pdf_router import PDFRouter
from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.executor.local import LocalPipelineExecutor

# Sample PDFs to test routing - use threshold analysis samples
SAMPLE_PDFS = [
    # Low OCR probability samples (should route to text_extraction)
    "examples_local/threshold_analysis/samples/low_ocr/low_ocr_01_<urn:uuid:449f2fe2-49b5-4609-a4c9-901ebbffbb81>.pdf",
    "examples_local/threshold_analysis/samples/low_ocr/low_ocr_02_<urn:uuid:12fcdb36-1e9d-4192-88c8-55a70ec2872f>.pdf",
    "examples_local/threshold_analysis/samples/low_ocr/low_ocr_03_<urn:uuid:ead811e4-4126-4ef9-8525-38beb86665a4>.pdf",
    # High OCR probability samples (should route to ocr_extraction)
    "examples_local/threshold_analysis/samples/high_ocr/high_ocr_01_<urn:uuid:98e53922-1ff8-45fd-be5c-41d9f906e869>.pdf",
    "examples_local/threshold_analysis/samples/high_ocr/high_ocr_02_<urn:uuid:f808a467-bd86-4c90-9e50-eeb5d47d36b5>.pdf",
    "examples_local/threshold_analysis/samples/high_ocr/high_ocr_03_<urn:uuid:3c02344a-24d1-4e38-961f-8b1f7bee9e32>.pdf",
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
            JsonlWriter(CLASSIFIED_OUTPUT, save_media_bytes=True),  # Save Media objects with PDF bytes
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
