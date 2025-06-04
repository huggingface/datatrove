import base64
import fitz  # PyMuPDF
from typing import Optional
import sys

def render_page_to_base64png_pymupdf(page, resize_longest_side_pixels: Optional[int], max_visual_tokens: int) -> str:
    # Get page dimensions
    rect = page.rect
    page_width = rect.width
    page_height = rect.height

    if page_width == 0 or page_height == 0:  # Avoid division by zero for empty pages
        zoom = 0
    else:
        # This is qwen model max image size
        # Qwen2.5 VL uses patches of 28x28 pixels. this ensures we are under 4096 visual tokens
        IMAGE_FACTOR = 28
        max_total_pixels = max_visual_tokens * IMAGE_FACTOR * IMAGE_FACTOR

        if resize_longest_side_pixels is not None:
            scale_factor = resize_longest_side_pixels / max(page_width, page_height)
            aligned_width = page_width * scale_factor if scale_factor > 1 else page_width
            aligned_height = page_height * scale_factor if scale_factor > 1 else page_height
        else:
            aligned_width = page_width
            aligned_height = page_height
        
        # Round to factor alignment (divisible by 28)
        aligned_width = max(IMAGE_FACTOR, round(aligned_width / IMAGE_FACTOR) * IMAGE_FACTOR)
        aligned_height = max(IMAGE_FACTOR, round(aligned_height / IMAGE_FACTOR) * IMAGE_FACTOR)
        
        # Check if aligned dimensions exceed max pixels and adjust if needed
        if aligned_width * aligned_height > max_total_pixels:
            # Scale down while maintaining factor alignment
            scale_factor = (max_total_pixels / (aligned_width * aligned_height)) ** 0.5
            aligned_width = max(IMAGE_FACTOR, int(aligned_width * scale_factor // IMAGE_FACTOR) * IMAGE_FACTOR)
            aligned_height = max(IMAGE_FACTOR, int(aligned_height * scale_factor // IMAGE_FACTOR) * IMAGE_FACTOR)
        
        # Calculate zoom factors for each dimension
        zoom_x = aligned_width / page_width
        zoom_y = aligned_height / page_height
        
        # Use the smaller zoom to maintain aspect ratio and ensure we don't exceed limits
        zoom = min(zoom_x, zoom_y)

        # Print to stderr the zoom factor and the page dimensions
        # print(f"Resize longest side: {resize_longest_side_pixels}px, Zoom factor: {zoom}, Page dimensions: {page_width}x{page_height}", file=sys.stderr)
        # print(f"Target aligned dimensions: {int(page_width * zoom)}x{int(page_height * zoom)}", file=sys.stderr)

    # Create the transformation matrix (same zoom for both dimensions to maintain aspect ratio)
    matrix = fitz.Matrix(zoom, zoom)

    # Render the page to a pixmap
    pix = page.get_pixmap(matrix=matrix)

    # Get PNG image bytes
    img_bytes = pix.tobytes("png")

    # Base64 encode the PNG bytes
    return base64.b64encode(img_bytes).decode("utf-8")