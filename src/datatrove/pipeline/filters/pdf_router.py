"""PDF Router Filter - Routes PDFs based on OCR probability prediction."""

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.media.predictor.scanned_pdf_predictor import PDFScannedPredictor
from datatrove.utils.typeshelper import StatHints


class PDFRouter(PipelineStep):
    """Route PDFs based on OCR probability to different processing paths.

    Uses XGBoost classifier to predict OCR probability and tags documents
    for either text extraction (low OCR) or OCR extraction (high OCR).

    Args:
        model_path: Path to trained XGBoost model file
        threshold: OCR probability threshold for routing (default: 0.5)
        low_ocr_tag: Tag for documents below threshold (default: "text_extraction")
        high_ocr_tag: Tag for documents above threshold (default: "ocr_extraction")
        num_pages_to_sample: Number of pages to sample for feature extraction (default: 8)
    """

    type = "🔀 - ROUTER"

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        low_ocr_tag: str = "text_extraction",
        high_ocr_tag: str = "ocr_extraction",
        num_pages_to_sample: int = 8,
    ):
        super().__init__()
        self.predictor = PDFScannedPredictor(
            path_to_model=model_path,
            num_pages_to_sample=num_pages_to_sample
        )
        self.threshold = threshold
        self.low_ocr_tag = low_ocr_tag
        self.high_ocr_tag = high_ocr_tag

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Route PDFs by predicting OCR probability and tagging with processing route.

        Args:
            data: Pipeline of Document objects with PDF content
            rank: Process rank for distributed processing
            world_size: Total number of processes

        Yields:
            Document objects with added metadata:
                - ocr_probability: float prediction from XGBoost model
                - processing_route: tag indicating extraction method
        """
        for doc in data:
            with self.track_time():
                self.stat_update(StatHints.total)

                # Get PDF bytes from Media object
                if not doc.media:
                    self.stat_update("no_media")
                    doc.metadata["prediction_error"] = "No media objects found"
                    continue

                if not doc.media[0].media_bytes:
                    self.stat_update("no_media_bytes")
                    doc.metadata["prediction_error"] = "Media object has no bytes"
                    continue

                pdf_bytes = doc.media[0].media_bytes

                # Get prediction from XGBoost classifier
                prediction = self.predictor.predict(pdf_bytes)

                # Check for prediction failure
                if "prediction_failed" in prediction:
                    self.stat_update("prediction_failed")
                    doc.metadata["prediction_error"] = prediction["prediction_failed"]
                    # Skip documents we can't classify
                    continue

                # Get OCR probability
                ocr_prob = prediction.get("ocr_prob", 0.0)

                # Route based on threshold
                processing_route = (
                    self.high_ocr_tag if ocr_prob >= self.threshold
                    else self.low_ocr_tag
                )

                # Add routing metadata
                doc.metadata["ocr_probability"] = ocr_prob
                doc.metadata["processing_route"] = processing_route

                # Add additional useful metadata from prediction
                doc.metadata["is_form"] = prediction.get("is_form", False)
                doc.metadata["garbled_text_ratio"] = prediction.get("garbled_text_ratio", 0.0)
                doc.metadata["is_encrypted"] = prediction.get("is_encrypted", False)
                doc.metadata["needs_password"] = prediction.get("needs_password", False)
                doc.metadata["num_pages"] = prediction.get("num_pages", 0)

                # Update stats
                self.stat_update(f"routed_to_{processing_route}")
                self.stat_update(StatHints.forwarded)
                self.update_doc_stats(doc)

                yield doc
