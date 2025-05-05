from typing import get_args

from datatrove.data import Document
import pymupdf
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from typing import List, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import binary_dilation, label, find_objects
from collections import Counter
import random
from statistics import mean

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS  – feel free to tweak                                             
# ──────────────────────────────────────────────────────────────────────────────
# CC-PDF rules
CC_PDF_VISIBLE_TEXT_THRESHOLD = 100
CC_PDF_IMAGE_COUNT_THRESHOLD = 0
CC_PDF_INVISIBLE_TEXT_THRESHOLD = 0

# Scanning limits
SCAN_MAX_PAGE                = 50        # how many pages to inspect at most
BITMAP_COVERAGE_TRESHOLD     = 0.75
BITMAP_PAGE_RATIO            = 0.50

# Text‑based rules
TEXT_THRESHOLD               = 100       # chars on a sampled page
AVG_TEXT_THRESHOLD           = 100       # mean chars per page
SAMPLE_RATIO                 = 0.10      # 10 % of pages, min below
MIN_SAMPLE_PAGES             = 5

# Image‑area rule
BIG_IMAGE_AREA_RATIO         = 0.50      # >50 % of page area counts as big
BIG_IMAGE_PAGE_RATIO         = 0.50      # >=50 % such pages ⇒ scan
REPEAT_ID_THRESHOLD          = 2         # obj_id seen ≥2 → transparent overlay

# Image‑count rule
JUNK_IMG_COUNT_MIN           = 10        # “many images” threshold
IMG_COUNT_EQUAL_RATIO        = 0.80      # 80 % pages with same img count

# Narrow‑strip rule
STRIP_LONG_EDGE_MULTIPLE     = 4         # long edge ≥ 4 × short edge
STRIP_IMAGE_RATIO            = 0.80      # 80 % of images are strips
STRIP_PAGE_RATIO             = 0.50      # on >=50 % of pages
PAGE_EDGE_COVER              = 0.90      # strip must cover 90 % width/height

# Merge tolerance when stitching split scans
MERGE_MAX_OFFSET             = 5         # pts
MERGE_MAX_GAP                = 2         # pts

# Invalid‑char rule
INVALID_CHAR_RATIO           = 0.01      # >15 % control chars ⇒ likely binary

# simple struct so the helper can live outside the Page class
@dataclass
class BoundingBox:
    l: int
    t: int
    r: int
    b: int


class PDFScanMinerUStats(BaseStats):
    name = "🎼 PDF Scan Miner U Stats"

    def __init__(
        self,
        output_folder: DataFolderLike,
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        histogram_round_digits: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)

    def cc_pdf_stats(self, pymupdf_doc) -> dict[str, int | float]:
        visible_len = 0
        hidden_len = 0
        img_count = 0
        for page in pymupdf_doc:
            # visible text
            visible_len += len(page.get_text("text"))
            # hidden text  (content in 'bbox==0' or clipped / invisible layers)
            for span in page.get_texttrace():          # fast, flat structure
                # Basic invisible checks -------------------------
                inv = span["type"] == 3 or span["opacity"] == 0
                inv = inv or span.get("layer")
                # Optional: check if later paint operations cover this span
                if not inv:
                    for other in page.get_texttrace():
                        if other["seqno"] > span["seqno"] and pymupdf.Rect(other["bbox"]).intersects(span["bbox"]):
                            inv = True
                            break
                if inv:
                    hidden_len += len(span["chars"])        # count hidden glyphs
            # images
            img_count += len(page.get_images())

        details = {
            "visible_text_len": visible_len,
            "hidden_text_len":  hidden_len,
            "image_count":      img_count,
        }
        return details


    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        """
        Returns a FLAT dict with:
        • global stats           (pages, image_pages, …)
        • is_text_pdf            (global decision)
        • <rule>_value           (metric each rule tested)
        • <rule>_decision        (True / False vote for that rule)
        """

        import pymupdf
        import io
        pdf_bytes = doc.media[0].media_bytes
        if not pdf_bytes:
            return {}
        try:
            pdf = pymupdf.open(None, io.BytesIO(pdf_bytes))
        except Exception as e:
            return {}

        total_page = len(pdf)
        if total_page == 0:
            return {}

        needs_password = pdf.needs_pass
        is_encrypted   = pdf.is_encrypted
        page_width     = pdf[0].rect.width
        page_height    = pdf[0].rect.height
        page_area      = page_width * page_height

        # 2 · collect per‑page data (≤ SCAN_MAX_PAGE)
        limit          = min(SCAN_MAX_PAGE, total_page)
        text_len_list  = []
        img_sz_list    = []
        img_num_list   = []
        objid_counter  = Counter()

        for page in pdf:
            text_len_list.append(len(page.get_text("text")))
            raw_imgs = pg.get_images()
            img_num_list.append(len(raw_imgs))
            page_imgs = []
            for img in raw_imgs:
                oid = img[0]
                objid_counter[oid] += 1
                rects = pg.get_image_rects(img, transform=True)
                if rects:
                    x0, y0, x1, y1 = map(int, rects[0][0])
                    if (x1 - x0) and (y1 - y0):
                        page_imgs.append((x0, y0, x1, y1, oid))
            img_sz_list.append(page_imgs)

        # 3 · helpers
        def merge_images(imgs):
            if not imgs:
                return []
            imgs.sort(key=lambda z: (z[1], z[0]))
            merged = [imgs[0]]
            for x0, y0, x1, y1, oid in imgs[1:]:
                lx0, ly0, lx1, ly1, _ = merged[-1]
                fw = abs(x1 - x0) >= PAGE_EDGE_COVER * page_width
                fh = abs(y1 - y0) >= PAGE_EDGE_COVER * page_height
                vert = (fw and abs(x0 - lx0) <= MERGE_MAX_OFFSET
                        and abs(x1 - lx1) <= MERGE_MAX_OFFSET
                        and abs(y0 - ly1) <= MERGE_MAX_GAP)
                hori = (fh and abs(y0 - ly0) <= MERGE_MAX_OFFSET
                        and abs(y1 - ly1) <= MERGE_MAX_OFFSET
                        and abs(x0 - lx1) <= MERGE_MAX_GAP)
                if vert or hori:
                    merged[-1] = (min(x0, lx0), min(y0, ly0),
                                max(x1, lx1), max(y1, ly1), oid)
                else:
                    merged.append((x0, y0, x1, y1, oid))
            return merged

        # 4 · rules – compute metric & decision
        flat = {}

        # image‑area rule
        repeat_ids = {oid for oid, c in objid_counter.items() if c >= REPEAT_ID_THRESHOLD}
        merged_pg  = [merge_images([im for im in pg if im[4] not in repeat_ids])
                    for pg in img_sz_list]
        big_pages = sum(
            (max((x1 - x0) * (y1 - y0) for x0, y0, x1, y1, _ in pg) / page_area
            > BIG_IMAGE_AREA_RATIO) if pg else False
            for pg in merged_pg
        )
        flat["by_image_area_value"]    = big_pages / limit
        flat["by_image_area_decision"] = flat["by_image_area_value"] < BIG_IMAGE_PAGE_RATIO

        # text‑len rule
        sample_n  = max(min(MIN_SAMPLE_PAGES, len(text_len_list)), int(len(text_len_list) * SAMPLE_RATIO))
        max_sample = max(text_len_list[i] for i in random.sample(range(len(text_len_list)), sample_n))
        flat["by_text_len_value"]    = max_sample
        flat["by_text_len_decision"] = max_sample > TEXT_THRESHOLD

        # avg‑words rule
        avg_words = (mean(text_len_list) if text_len_list else 0)
        flat["by_avg_words_value"]    = avg_words
        flat["by_avg_words_decision"] = avg_words > AVG_TEXT_THRESHOLD

        # img‑num rule
        top80     = img_num_list[:int(IMG_COUNT_EQUAL_RATIO * len(img_num_list))]
        identical = len(set(top80)) == 1 if top80 else False
        many_imgs = max(img_num_list, default=0) >= JUNK_IMG_COUNT_MIN
        non_empty = sum(bool(pg) for pg in img_sz_list)
        flat["by_img_num_non_empty_pages"] = non_empty
        flat["by_img_num_identical_counts"] = identical
        flat["by_img_num_max_img_count"] = max(img_num_list, default=0)
        flat["by_img_num_decision"] = not (non_empty <= 1 and identical and many_imgs)

        # narrow‑strip rule
        strip_pages = 0
        for pg in img_sz_list:
            if not pg:
                continue
            strips = 0
            for x0, y0, x1, y1, _ in pg:
                w, h = x1 - x0, y1 - y0
                if ((w >= PAGE_EDGE_COVER * page_width and w >= STRIP_LONG_EDGE_MULTIPLE * h) or
                    (h >= PAGE_EDGE_COVER * page_height and h >= STRIP_LONG_EDGE_MULTIPLE * w)):
                    strips += 1
            if strips >= 5 and strips / len(pg) >= STRIP_IMAGE_RATIO:
                strip_pages += 1
        flat["by_img_narrow_strip_value"]    = strip_pages / limit
        flat["by_img_narrow_strip_decision"] = flat["by_img_narrow_strip_value"] < STRIP_PAGE_RATIO

        # garbled text
        all_text = 0
        garbled_text = 0
        for page in pdf:
            page_text = page.get_text("text", pymupdf.TEXT_CID_FOR_UNKNOWN_UNICODE)
            all_text += len(page_text)
            garbled_text += page_text.count(chr(0xfffd))
        ratio_bad = garbled_text / all_text
        flat["by_invalid_chars_value"]    = ratio_bad
        flat["by_invalid_chars_decision"] = ratio_bad <= INVALID_CHAR_RATIO

        # 5 · global label
        is_text_pdf = all(v for k, v in flat.items() if k.endswith("_decision"))

        # CC-pdf
        cc_pdf_stats = self.cc_pdf_stats(pdf)
        # By size of invisible text
        flat["by_visible_text_len_value"] = cc_pdf_stats["visible_text_len"]
        flat["by_num_of_images_value"] = cc_pdf_stats["image_count"]
        flat["by_size_of_invisible_text_value"] = cc_pdf_stats["hidden_text_len"]
        flat["cc_pdf_decision"] = flat["by_visible_text_len_value"] >= CC_PDF_VISIBLE_TEXT_THRESHOLD and flat["by_num_of_images_value"] <= CC_PDF_IMAGE_COUNT_THRESHOLD and flat["by_size_of_invisible_text_value"] <= CC_PDF_INVISIBLE_TEXT_THRESHOLD
        


        # 6 · assemble final flat dict
        stats = {
            "pages":           total_page,
            "images":          sum(len(pg) for pg in img_sz_list),
            "image_pages":     sum(bool(pg) for pg in img_sz_list),
            "is_encrypted":    is_encrypted,
            "needs_password":  needs_password,
            "is_text_pdf":     is_text_pdf,
        }
        stats.update(flat)
        return stats