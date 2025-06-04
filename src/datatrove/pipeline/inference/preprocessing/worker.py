import sys
import json
import base64
import io
import zstandard as zstd
import warnings
from datatrove.pipeline.inference.utils.page_rendering import render_page_to_base64png_pymupdf

import pymupdf
warnings.filterwarnings('ignore')
# Mute pymupdf logs
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)

def set_oom_score_adj(score):
    """Set OOM score adjustment for the current process."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(f"{score}\n")
    except (FileNotFoundError, PermissionError):
        pass

def process_pdf_pages(zstd_data_b64, length, resize_longest_side_pixels, max_visual_tokens, image_rotation, id):
    """Process PDF pages and return results."""
    # Set OOM score
    
    # Decode base64 data
    pymupdf_doc = None
    try:
        zstd_data = base64.b64decode(zstd_data_b64)
        
        # Decompress PDF data
        zstd_decompressor = zstd.ZstdDecompressor(format=zstd.FORMAT_ZSTD1_MAGICLESS)
        with zstd_decompressor.stream_reader(io.BytesIO(zstd_data), read_across_frames=False, closefd=False) as zstd_stream_reader:
            file_bytes = zstd_stream_reader.read(length)

        # Save to ./pdfs/
        # os.makedirs(f"{OUTPUT_FOLDER}/{id}", exist_ok=True)
        # with open(f"{OUTPUT_FOLDER}/{id}/document.pdf", "wb") as f:
        #     f.write(file_bytes)
        
        # Open PDF document
        pymupdf_doc = pymupdf.open("pdf", file_bytes)
        num_pages = len(pymupdf_doc)
    
        # Send number of pages first
        print(json.dumps({"type": "num_pages", "data": num_pages}), flush=True)
        
        # Process each page
        for page_num in range(num_pages):
            try:
                page = pymupdf_doc[page_num]
                # Render page to base64 PNG
                image_base64 = render_page_to_base64png_pymupdf(page, resize_longest_side_pixels=resize_longest_side_pixels, max_visual_tokens=max_visual_tokens)
                
                # Apply rotation if needed
                # if image_rotation != 0:
                #     image_bytes = base64.b64decode(image_base64)
                #     with Image.open(BytesIO(image_bytes)) as img:
                #         rotated_img = img.rotate(-image_rotation, expand=True)
                #         buffered = BytesIO()
                #         rotated_img.save(buffered, format="PNG")
                #     image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Send page result
                result = {
                    "type": "page",
                    "data": {
                        "page_num": page_num,
                        "page_image_b64": image_base64,
                        # TODO: Possible implement page text for olmo-ocr
                        "page_text": "",
                    }
                }
                print(json.dumps(result), flush=True)
                
            except Exception as e:
                error_result = {
                    "type": "error",
                    "data": {
                        "page_num": page_num,
                        "error": str(e)
                    }
                }
                print(json.dumps(error_result), flush=True)
                
        # Signal completion
        print(json.dumps({"type": "complete", "data": None}), flush=True)
        
    finally:
        if pymupdf_doc is not None:
            pymupdf_doc.close()



if __name__ == "__main__":
    set_oom_score_adj(1000)
    while True:
        input_data = json.loads(sys.stdin.readline())
        process_pdf_pages(
            input_data["zstd_data_b64"],
            input_data["length"],
            input_data["resize_longest_side_pixels"],
            input_data["max_visual_tokens"],
            input_data["image_rotation"],
            input_data["id"]
        )