from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.typeshelper import StatHints


class PDFTruncationFilter(BaseFilter):
    """Filter for handling truncated PDFs from CommonCrawl.

    Can either exclude truncated PDFs or mark them for re-fetching.

    Args:
        action: Action to take on truncated PDFs
            - "exclude": Remove truncated PDFs from pipeline
            - "mark_for_refetch": Keep but mark for URL re-fetching
            - "include": Keep all PDFs regardless of truncation
        check_pdf_header: Also check if PDF content starts with valid header
        min_size_bytes: Minimum size in bytes to consider non-truncated (for older dumps)
    """

    name = "📄 PDF Truncation"

    def __init__(
        self,
        action: str = "exclude",
        check_pdf_header: bool = True,
        min_size_bytes: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        if action not in ["exclude", "mark_for_refetch", "include"]:
            raise ValueError(f"Invalid action: {action}. Must be 'exclude', 'mark_for_refetch', or 'include'")

        self.action = action
        self.check_pdf_header = check_pdf_header
        self.min_size_bytes = min_size_bytes

    def filter(self, doc: Document) -> bool | Document:
        """Filter truncated PDFs based on configured action."""

        # Check if document is marked as truncated
        is_truncated = doc.metadata.get("is_truncated", False)
        truncation_reason = doc.metadata.get("truncation_reason")

        # Additional checks for truncation
        additional_truncation_reasons = []

        if self.check_pdf_header and isinstance(doc.text, bytes):
            # Check for valid PDF header
            if not doc.text.startswith(b'%PDF-'):
                additional_truncation_reasons.append("invalid_pdf_header")
                is_truncated = True

        # Check minimum size
        content_length = doc.metadata.get("content_length", len(doc.text) if hasattr(doc.text, '__len__') else 0)
        if content_length < self.min_size_bytes:
            additional_truncation_reasons.append("too_small")
            is_truncated = True

        # Update metadata with additional findings
        if additional_truncation_reasons:
            existing_reasons = [truncation_reason] if truncation_reason else []
            all_reasons = existing_reasons + additional_truncation_reasons
            doc.metadata["truncation_reason"] = ";".join(filter(None, all_reasons))
            doc.metadata["is_truncated"] = True

        # Take action based on configuration
        if not is_truncated:
            # Not truncated - always include
            self.stat_update("not_truncated")
            return True

        # Handle truncated PDFs
        if self.action == "exclude":
            self.stat_update("excluded_truncated")
            return False

        elif self.action == "mark_for_refetch":
            # Mark for re-fetching and include
            doc.metadata["needs_refetch"] = True
            doc.metadata["original_content_length"] = content_length
            self.stat_update("marked_for_refetch")
            return True

        elif self.action == "include":
            # Include all regardless of truncation
            self.stat_update("included_truncated")
            return True

        return True


class PDFValidationFilter(BaseFilter):
    """Validate PDF content quality and structure.

    More comprehensive validation beyond just truncation detection.

    Args:
        check_pdf_structure: Validate basic PDF structure
        min_pages: Minimum number of pages (if detectable)
        max_corruption_ratio: Maximum ratio of corrupted bytes to accept
    """

    name = "🔍 PDF Validation"

    def __init__(
        self,
        check_pdf_structure: bool = True,
        min_pages: int = 1,
        max_corruption_ratio: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.check_pdf_structure = check_pdf_structure
        self.min_pages = min_pages
        self.max_corruption_ratio = max_corruption_ratio

    def filter(self, doc: Document) -> bool | Document:
        """Validate PDF content quality."""

        if not isinstance(doc.text, bytes):
            self.stat_update("not_bytes")
            return False

        # Basic PDF header check
        if not doc.text.startswith(b'%PDF-'):
            self.stat_update("invalid_header")
            return False

        validation_issues = []

        if self.check_pdf_structure:
            # Check for PDF trailer
            if b'trailer' not in doc.text.lower():
                validation_issues.append("missing_trailer")

            # Check for xref table
            if b'xref' not in doc.text.lower():
                validation_issues.append("missing_xref")

            # Check for basic PDF objects
            if b'obj' not in doc.text.lower():
                validation_issues.append("missing_objects")

        # Check corruption ratio (rough heuristic)
        if len(doc.text) > 100:  # Only for docs with reasonable size
            # Count potential corruption indicators
            corruption_indicators = doc.text.count(b'\x00') + doc.text.count(b'\xff\xff')
            corruption_ratio = corruption_indicators / len(doc.text)

            if corruption_ratio > self.max_corruption_ratio:
                validation_issues.append(f"high_corruption_ratio_{corruption_ratio:.3f}")

        # Update metadata with validation results
        if validation_issues:
            doc.metadata["pdf_validation_issues"] = ";".join(validation_issues)
            doc.metadata["pdf_validation_failed"] = True
            self.stat_update("validation_failed")
            return False

        else:
            doc.metadata["pdf_validation_passed"] = True
            self.stat_update("validation_passed")
            return True