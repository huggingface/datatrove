import sys
import json
import base64
import io
import zstandard as zstd
import warnings
from datatrove.pipeline.inference.utils.page_rendering import render_page_to_base64png_pymupdf
from fasttext.FastText import _FastText
from datatrove.io import cached_asset_path_or_download

import pymupdf
warnings.filterwarnings('ignore')
# Mute pymupdf logs
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)

# Path to your fastText language identification model
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# Download/cached model path
fasttext_model_path = cached_asset_path_or_download(
    FASTTEXT_MODEL_URL, namespace="filters", subfolder="fasttext", desc="fast-text model"
)
fasttext_model = _FastText(fasttext_model_path)

def set_oom_score_adj(score):
    """Set OOM score adjustment for the current process."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(f"{score}\n")
    except (FileNotFoundError, PermissionError):
        pass

zstd_decompressor = zstd.ZstdDecompressor(format=zstd.FORMAT_ZSTD1_MAGICLESS)

def predict_language(text):
    labels, scores = fasttext_model.predict(text.replace('\n', ' '), k=1)
    if labels and scores and scores[0] >= 0.65:
        return labels[0].replace('__label__', ''), scores[0]
    return None, None

def process_pdf_pages(data_b64, is_zstd_compressed, length, resize_longest_side_pixels, resize_longest_side_pixels_en, max_visual_tokens, image_rotation, id):
    """Process PDF pages and return results."""
    # Set OOM score
    
    # Decode base64 data
    pymupdf_doc = None
    try:
        file_bytes = base64.b64decode(data_b64)

        # Decompress PDF data
        if is_zstd_compressed:
            with zstd_decompressor.stream_reader(io.BytesIO(file_bytes), read_across_frames=False, closefd=False) as zstd_stream_reader:
                file_bytes = zstd_stream_reader.read(length)

        # Open PDF document
        pymupdf_doc = pymupdf.open("pdf", file_bytes)
        num_pages = len(pymupdf_doc)

        # --- Language detection logic ---
        all_text = ""
        for page_num in range(num_pages):
            try:
                page = pymupdf_doc[page_num]
                all_text += page.get_text()
            except Exception:
                continue
        avg_chars_per_page = len(all_text) / num_pages if num_pages > 0 else 0
        if avg_chars_per_page >= 50:
            language, confidence = predict_language(all_text)
        else:
            language, confidence = None, None
        # --- End language detection logic ---

        # Send number of pages and language first
        print(json.dumps({"type": "num_pages", "data": {"num_pages": num_pages, "language": language}}), flush=True)
        
        # Use max_visual_tokens=1280 if language is "en"
        # if language == "en":
        #     resize_longest_side_pixels = resize_longest_side_pixels_en
            # print(f"Language is {language}, using resize_longest_side_pixels = {resize_longest_side_pixels}", flush=True, file=sys.stderr)

        # Process each page
        for page_num in range(num_pages):
            try:
                page = pymupdf_doc[page_num]
                # Render page to base64 PNG
                image_base64 = render_page_to_base64png_pymupdf(
                    page,
                    resize_longest_side_pixels=resize_longest_side_pixels,
                    max_visual_tokens=max_visual_tokens
                )
                
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
            input_data["data_b64"],
            input_data["is_zstd_compressed"],
            input_data["length"],
            input_data["resize_longest_side_pixels"],
            input_data["resize_longest_side_pixels_en"],
            input_data["max_visual_tokens"],
            input_data["image_rotation"],
            input_data["id"]
        )