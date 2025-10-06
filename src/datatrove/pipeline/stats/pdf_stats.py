from typing import get_args
from datatrove.pipeline.stats.base import BaseStats
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.config import GROUP
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, TopKConfig

from datatrove.data import Document, MediaType, Media
from datatrove.utils import stats

class PdfStats(BaseStats):
    """
    Summary stats of line level metrics.
    """

    name = "📄 PDF stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        histogram_round_digits: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
        use_chars_for_histogram: bool = False,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config, use_chars_for_histogram)


    # Global document stats
    def get_metadata_stats(self, doc: Document) -> dict[str, int | float | None]:
        return {
            "pdf_version": doc.metadata.get("format", None),
            "encryption": doc.metadata.get("encryption", None),
            "producer_software": doc.metadata.get("producer", None),
        }

    def page_count(self, pymupdf_doc) -> int:
        """
        Returns the number of pages in the PDF document.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            int: Number of pages in the document.
        """
        return len(pymupdf_doc)

    def image_count(self, pymupdf_doc) -> int:
        """
        Returns the total number of images in the PDF document.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            int: Number of images in the document.
        """
        total_images = 0
        
        for page_num in range(len(pymupdf_doc)):
            page = pymupdf_doc[page_num]
            # Get all images on the page
            image_list = page.get_images()
            total_images += len(image_list)
        
        return total_images

    def object_count(self, pymupdf_doc) -> int:
        """
        Returns the total number of PDF objects in the document.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            int: Number of PDF objects.
        """
        # Get the xref table which contains all objects
        xref = pymupdf_doc.xref_length()
        return xref - 1  # Subtract 1 because xref[0] is the free object list

    def hyperlink_count(self, pymupdf_doc) -> int:
        """
        Returns the total number of hyperlinks in the PDF document.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            int: Number of hyperlinks in the document.
        """
        total_links = 0
        
        for page_num in range(len(pymupdf_doc)):
            page = pymupdf_doc[page_num]
            # Get all links on the page
            links = page.get_links()
            total_links += len(links)
        
        return total_links

    def form_field_count(self, pymupdf_doc) -> int:
        """
        Returns the total number of form fields in the PDF document.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            int: Number of form fields in the document.
        """
        # PyMuPDF provides access to form fields via get_widget_list
        # This is the correct method to use instead of get_form_fields
        try:
            # First, check if the document has a form
            if not pymupdf_doc.is_form_pdf:
                return 0
                
            # Get all widgets (form fields) in the document
            widget_count = 0
            for page in pymupdf_doc:
                widget_count += len(list(page.widgets()))
                
            return widget_count
        except:
            # In case of any errors, return 0
            return 0

    def toc_count(self, pymupdf_doc) -> int:
        """
        Returns the number of table of contents entries (outlines/bookmarks) in the PDF document.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            int: Number of TOC entries in the document, 0 if none.
        """
        toc = pymupdf_doc.get_toc()
        return len(toc) if toc else 0

    def multi_column_page_count(self, pymupdf_doc) -> int:
        """
        Estimates the number of pages with multiple columns of text using
        histogram analysis of text block positions.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            int: Count of pages with multiple text columns, 0 if none detected.
        """
        # Import necessary libraries
        import numpy as np
        from scipy import ndimage, signal
        
        multi_column_count = 0
        total_pages = len(pymupdf_doc)
        for page_num in range(total_pages):
            page = pymupdf_doc[page_num]
            
            # Get text blocks
            blocks = page.get_text("dict")["blocks"]
            
            # Extract text block positions (only blocks with text)
            text_blocks = []
            for block in blocks:
                if "lines" in block and len(block["lines"]) > 0:
                    text_blocks.append(block)
            
            # Skip pages with too few text blocks for column analysis
            if len(text_blocks) < 4:
                continue
                
            # Extract left (x-min) positions of text blocks
            left_values = [block["bbox"][0] for block in text_blocks]
            
            # Try to detect columns using histogram analysis
            try:
                # Create histogram of left positions
                num_bins = min(10, len(left_values) // 3 + 1)  # Adjust bin count based on data amount
                hist, bin_edges = np.histogram(left_values, bins=num_bins)
                
                # Apply Gaussian smoothing to the histogram
                kernel_width = 1
                hist = ndimage.gaussian_filter1d(hist, kernel_width)
                
                # Add padding to the histogram for edge handling
                min_val = min(hist)
                hist = np.insert(hist, [0, len(hist)], min_val)
                bin_width = bin_edges[1] - bin_edges[0]
                bin_edges = np.insert(bin_edges, [0, len(bin_edges)], 
                                      [bin_edges[0] - bin_width, bin_edges[-1] + bin_width])
                
                # Find peaks in the histogram
                min_prominence = 0.3
                peaks, _ = signal.find_peaks(hist, prominence=min_prominence * np.max(hist))
                
                # If we have multiple peaks, it suggests multiple columns
                if len(peaks) > 1:
                    # Calculate derivatives to find separators
                    derivatives = np.diff(hist)
                    separators = []
                    
                    for i in range(len(peaks)-1):
                        peak_left = peaks[i]
                        peak_right = peaks[i+1]
                        max_deriv_index = np.argmax(derivatives[peak_left:peak_right]) + peak_left
                        separator_x = bin_edges[max_deriv_index + 1]
                        separators.append(separator_x)
                    
                    # If we found separators, this is a multi-column page
                    if len(separators) > 0:
                        multi_column_count += 1

                            
            except Exception:
                pass
        
        return multi_column_count

    def text_coverage_per_page(self, pymupdf_doc) -> list[float]:
        """
        Returns the maximum ratio of text-covered area to total page area across all pages.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            float: Maximum text coverage ratio (0.0 to 1.0).
        """
        ratios = []
        for page_num in range(len(pymupdf_doc)):
            page = pymupdf_doc[page_num]
            # Get page dimensions
            page_width = page.rect.width
            page_height = page.rect.height
            total_area = page_width * page_height
            
            if total_area == 0:
                continue
                
            # Get text blocks
            blocks = page.get_text("dict")["blocks"]
            
            # Calculate total text area
            text_area = 0.0
            for block in blocks:
                if "lines" in block and len(block["lines"]) > 0:
                    # Get block dimensions
                    bbox = block["bbox"]
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    text_area += width * height
            
            # Calculate ratio for this page
            ratio = text_area / total_area
            ratios.append(ratio)
        return ratios




    def is_digital_born(self, stats) -> bool:
        """
        Determines if a PDF is likely to be digitally created (rather than scanned).
        
        Args:
            stats: Dictionary of PDF statistics.
            
        Returns:
            bool: True if the PDF is likely digital-born, False otherwise.
        """
        # https://arxiv.org/pdf/2304.14953
        # 1. Has significant amount of selectable text
        if 'visible_text_length' in stats and stats['visible_text_length'] > 100:
            return True

        # 2. Has 0 images
        if 'image_count' in stats and stats['image_count'] == 0:
            return True

        # 3. Has no hidden text
        if 'hidden_text_length' in stats and stats['hidden_text_length'] == 0:
            return True

        # If none of the above conditions are met, likely not digital-born
        return False

    
    def is_digital_born_per_page(self, covered_area_ratio_per_page, text_per_page) -> list[bool]:
        """
        Determines if each page of a PDF is likely to be digitally created (rather than scanned).
        
        Args:
            stats: Dictionary of PDF statistics.
            
        Returns:
            list[bool]: List of booleans indicating if each page is likely digital-born.
        """

        is_digital_born_per_page = []
        for i in range(len(covered_area_ratio_per_page)):
            if covered_area_ratio_per_page[i] > 0.85 and text_per_page[i] > 5:
                is_digital_born_per_page.append(True)
            else:
                is_digital_born_per_page.append(False)

        return is_digital_born_per_page


    def text_based_stats(self, pymupdf_doc) -> tuple[dict[str, int | float], list[int]]:
        """
        Computes text-based statistics for the PDF document.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            tuple[dict[str, int | float], list[int]]: Tuple containing text-related statistics and visible text per page.
        """
        stats = {}
        
        # 1. Calculate visible text length
        visible_text = 0
        hidden_text = 0
        visible_text_per_page = []
        
        for page_num in range(len(pymupdf_doc)):
            page = pymupdf_doc[page_num]
            
            # Get visible text
            visible_text += len(page.get_text("text"))
            
            # Get hidden text (OCR layer)
            raw_text = page.get_text("rawdict")
            raw_text_count = 0
            if 'blocks' in raw_text:
                for block in raw_text['blocks']:
                    if 'lines' in block:
                        for line in block['lines']:
                            if 'spans' in line:
                                for span in line['spans']:
                                    if 'text' in span:
                                        raw_text_count += len(span['text'])
            
            # Approximate hidden text by comparing raw text with visible text
            page_visible = len(page.get_text("text"))
            hidden_text += max(0, raw_text_count - page_visible)
            visible_text_per_page.append(page_visible)

        stats['visible_text_length'] = visible_text
        stats['hidden_text_length'] = hidden_text
            
        return stats, visible_text_per_page

    def is_tagged(self, pymupdf_doc) -> bool:
        """
        Checks if the PDF document has a tagged structure (important for accessibility).
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            bool: True if the document is tagged, False otherwise.
        """
        # In PyMuPDF, we can check if the document is tagged using get_xml_metadata()
        # If the document has XML metadata with tags, it's likely a tagged PDF
        try:
            xml_metadata = pymupdf_doc.get_xml_metadata()
            return xml_metadata is not None and len(xml_metadata) > 0
        except:
            # If there's an error accessing XML metadata, assume it's not tagged
            return False
            
            
    def get_document_language(self, pymupdf_doc) -> str:
        """
        Returns the primary language of the PDF document as specified in metadata.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            str: Language code (e.g., 'en-US', 'fr'), empty string if not specified.
        """
        return pymupdf_doc.language

    def compute_image_coverage(self, pymupdf_doc) -> dict[str, float]:
        """
        Computes the average image coverage ratio across all pages.
        High coverage (>0.9) is typical for scanned documents.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            dict[str, float]: Dictionary with image coverage statistics
        """
        total_coverage = 0.0
        high_coverage_pages = 0
        total_pages = len(pymupdf_doc)
        coverage_per_page = []

        if total_pages == 0:
            return {
                'avg_image_coverage': 0.0,
                'high_coverage_pages': 0,
            }
            
        for page_num in range(total_pages):
            page = pymupdf_doc[page_num]
            images = page.get_images(full=True)
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height
            
            if page_area == 0:
                continue
                
            max_coverage = 0.0
            if images:
                for img in images:
                    image_rect = page.get_image_bbox(img)
                    if image_rect:
                        # Calculate intersection area
                        intersection = image_rect & page_rect
                        intersection_area = intersection.width * intersection.height
                        coverage = intersection_area / page_area
                        max_coverage = max(max_coverage, coverage)
            
            total_coverage += max_coverage
            if max_coverage >= 0.8:
                high_coverage_pages += 1
            coverage_per_page.append(max_coverage)
        return {
            'avg_image_coverage': total_coverage / total_pages,
            'high_coverage_pages': high_coverage_pages,
        }
        
        
    def analyze_text_as_drawings(self, pymupdf_doc) -> dict[str, int | float]:
        """
        Analyzes text rendered as drawings, which is common in some OCR'd PDFs.
        
        Args:
            pymupdf_doc: The PyMuPDF document object.
            
        Returns:
            dict: Statistics about text rendered as drawings
        """
        pages_with_many_paths = 0
        pages_with_text_drawings = 0
        total_pages = len(pymupdf_doc)
        
        for page_num in range(total_pages):
            page = pymupdf_doc[page_num]
            try:
                drawings = page.get_drawings()
                if len(drawings) > 1000:  # Many paths often indicates OCR'd text as drawings
                    pages_with_many_paths += 1
                    
                    # Count small fill paths with black color (typical for text)
                    small_text_like_paths = 0
                    for drawing in drawings:
                        for item in drawing["items"]:
                            if item["type"] == "f":  # Fill path
                                # Check for small rectangles (character-sized)
                                if "rect" in item:
                                    rect = item["rect"]
                                    width = rect[2] - rect[0]
                                    height = rect[3] - rect[1]
                                    
                                    # Character-like dimensions and black fill color
                                    if 1 <= width <= 30 and 5 <= height <= 40:
                                        if "color" in item and item["color"] == (0, 0, 0):
                                            small_text_like_paths += 1
                    
                    # If many small text-like paths, likely text rendered as drawings
                    if small_text_like_paths > 100:
                        pages_with_text_drawings += 1
            except:
                pass
                
        return {
            'pages_with_many_paths': pages_with_many_paths,
            'pages_with_text_drawings': pages_with_text_drawings,
            'text_drawing_page_ratio': pages_with_text_drawings / total_pages if total_pages > 0 else 0.0
        }

    def is_ocr(self, stats) -> bool:
        """
        Determines if a PDF contains OCR'd text based on precomputed statistics.
        
        Args:
            stats: Dictionary of precomputed PDF statistics.
            
        Returns:
            bool: True if the PDF appears to be OCR'd, False otherwise.
        """
        # Count how many OCR indicators we have
        # High image coverage is a strong indicator of scanned content
        if 'high_coverage_pages' in stats and stats['high_coverage_pages'] >=1:
            return True
            
        # Presence of OCR fonts is a very strong indicator
        # Text rendered as drawings is common in some OCR workflows
        if 'text_drawing_page_ratio' in stats and stats['text_drawing_page_ratio'] > 0.3:
            return True
            
        return False

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        """
        Extract stats from a document.
        Args:
            doc: The document to extract stats from.

        Returns:
            A dictionary of statistics, where the key is the stat name and the value is the stat value.
        """
        pdf_stats = {}
        for media in doc.media:
            if media.type == MediaType.DOCUMENT:
                pdf_stats = self.analyze_pdf(media)

        return pdf_stats

    def analyze_pdf(self, media: Media) -> dict[str, int | float]:
        """
        Analyzes a PDF document and returns various statistics.
        
        Args:
            media: Media object containing PDF data.
            
        Returns:
            dict: Dictionary with PDF statistics (numeric values only).
        """
        import pymupdf
        if media.media_bytes is None:
            return {}
        doc = pymupdf.open("pdf", media.media_bytes)
        
        per_page_text_coverage = self.text_coverage_per_page(doc)
        # Collect basic statistics
        stats = {
            'page_count': self.page_count(doc),
            'image_count': self.image_count(doc),
            'object_count': self.object_count(doc),
            'hyperlink_count': self.hyperlink_count(doc),
            'form_field_count': self.form_field_count(doc),
            'toc_count': self.toc_count(doc),
            'multi_column_page_count': self.multi_column_page_count(doc),
            'is_tagged': 1 if self.is_tagged(doc) else 0,
            'pages_with_sub_20_text_coverage': sum(1 for coverage in per_page_text_coverage if coverage < 0.2),
            "min_page_text_coverage": min(per_page_text_coverage),
            "average_page_text_coverage": sum(per_page_text_coverage) / len(per_page_text_coverage)
        }
        
        # Add text-based statistics
        text_stats, visible_text_per_page = self.text_based_stats(doc)
        stats.update(text_stats)
        stats.update({
            "sub_50_characters_per_page": sum(1 for text_len in visible_text_per_page if text_len < 50) / len(visible_text_per_page),
            "min_visible_text_per_page": min(visible_text_per_page),
        })
        
        # Compute OCR-related statistics
        image_coverage_stats = self.compute_image_coverage(doc)
        stats.update(image_coverage_stats)
        
        drawing_stats = self.analyze_text_as_drawings(doc)
        stats.update(drawing_stats)
        
        # Determine if document is OCR'd based on precomputed stats
        stats['is_ocr'] = 1 if self.is_ocr(stats) else 0
        
        # Add digital-born flag based on collected stats
        stats['is_digital_born'] = 1 if self.is_digital_born(stats) else 0
        
        return stats

