"""Recover raster pictures the VLM layout pass missed.

The Unlimited-OCR layout model occasionally returns a `Table` entry whose cells
are documented in text only, dropping the embedded line drawings that lived
inside the cells (anchor schematics, parts diagrams, etc.). When the source
is a native PDF we can recover those images directly from the PDF: every
embedded raster has a known on-page rect, and any rect that lands inside a
Table bbox but isn't represented by a `Picture` entry can be re-attached.

Public surface
--------------
recover_missing_pictures(pages) — mutates each page dict in place, adding
synthesized `Picture` entries (with an `image_obj` PIL crop ready for the
downstream `process_pictures` cropper) for any embedded raster the VLM
skipped. Scanned PDFs and image inputs are no-ops — those have no embedded
raster info to recover from.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF — already a dependency via doc_processing
from PIL import Image


log = logging.getLogger("ocr_service")


MIN_PICTURE_SIDE_PT = 12.0


DEDUP_IOU_THRESHOLD = 0.4


def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (a_area + b_area - inter)


def _bbox_contains(outer: Tuple[float, float, float, float],
                   inner: Tuple[float, float, float, float],
                   slack: float = 2.0) -> bool:
    """True when `inner` sits inside `outer` (with `slack` px tolerance on
    each edge). Used to decide whether an embedded raster falls inside a
    Table bbox."""
    return (
        inner[0] >= outer[0] - slack
        and inner[1] >= outer[1] - slack
        and inner[2] <= outer[2] + slack
        and inner[3] <= outer[3] + slack
    )


def _extract_pdf_raster_bboxes(
    pdf_source: str, page_index: int,
) -> List[Tuple[float, float, float, float]]:
    """Per-raster on-page bbox in PDF points for `pdf_source` page
    `page_index`. Returns an empty list when the PDF can't be opened or the
    page has no embedded rasters."""
    rects: List[Tuple[float, float, float, float]] = []
    try:
        with fitz.open(pdf_source) as doc:
            if page_index >= len(doc):
                return []
            page = doc.load_page(page_index)
            for info in page.get_image_info():
                bbox = info.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = (float(v) for v in bbox)
                if (x2 - x1) < MIN_PICTURE_SIDE_PT:
                    continue
                if (y2 - y1) < MIN_PICTURE_SIDE_PT:
                    continue
                rects.append((x1, y1, x2, y2))
    except Exception as exc:
        log.warning("[recover] could not open %s: %s", pdf_source, exc)
        return []
    return rects


def _rect_pt_to_px(
    rect_pt: Tuple[float, float, float, float], zoom: float,
) -> Tuple[int, int, int, int]:
    return (
        int(round(rect_pt[0] * zoom)),
        int(round(rect_pt[1] * zoom)),
        int(round(rect_pt[2] * zoom)),
        int(round(rect_pt[3] * zoom)),
    )


def _existing_picture_rects_px(
    entries: List[Dict],
) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for e in entries:
        if e.get("category") != "Picture":
            continue
        bb = e.get("bbox")
        if not bb or len(bb) != 4:
            continue
        out.append(tuple(int(v) for v in bb))
    return out


def _is_inside_any_table(
    rect_px: Tuple[int, int, int, int], table_bboxes_px: List[Tuple[int, int, int, int]],
) -> bool:
    return any(_bbox_contains(t, rect_px) for t in table_bboxes_px)


def _is_duplicate_of_existing(
    rect_px: Tuple[int, int, int, int],
    existing_picture_rects: List[Tuple[int, int, int, int]],
) -> bool:
    return any(
        _iou(rect_px, e) >= DEDUP_IOU_THRESHOLD for e in existing_picture_rects
    )


def _next_entry_position(
    entries: List[Dict],
    new_bbox_px: Tuple[int, int, int, int],
) -> int:
    """Insertion index that keeps reading order — find the first entry whose
    bbox top sits below the new picture's top, fall back to end of list."""
    new_top = new_bbox_px[1]
    for i, e in enumerate(entries):
        bb = e.get("bbox") or [0, 0, 0, 0]
        if len(bb) >= 2 and bb[1] > new_top:
            return i
    return len(entries)


def recover_missing_pictures(pages: List[Dict]) -> List[Dict]:
    """For each page sourced from a native PDF, find embedded rasters inside
    Table bboxes that are missing from the VLM-emitted entries and inject a
    `Picture` entry per raster. Returns the same `pages` list, mutated in
    place — keeps the call-site shape identical to other pipeline steps.

    Each synthesized entry carries:
        - bbox: [x1, y1, x2, y2] in page-image pixels (same convention as
          every other entry)
        - category: "Picture"
        - text: ""
        - image_obj: PIL.Image cropped from the page's original_image (the
          same shape `process_pictures` later expects, so the downstream
          flow does not need to special-case recovered pictures)
        - source: "recovered" — provenance marker, useful for diagnostics
          and so the markdown serializer can choose where to place the
          image placeholder.
    """
    for page in pages:
        pdf_source: Optional[str] = page.get("pdf_source")
        if not pdf_source:
            # Scanned PDF page or raw image upload — no embedded raster
            # info to recover from.
            continue
        page_index: int = int(page.get("page_index", 0))
        zoom: float = float(page.get("zoom") or 1.0)
        original_img: Optional[Image.Image] = page.get("image") or page.get("original_image")
        if original_img is None:
            continue
        entries: List[Dict] = page.get("layout_result") or []
        if not entries:
            continue

        table_bboxes_px = [
            tuple(int(v) for v in (e.get("bbox") or [0, 0, 0, 0]))
            for e in entries
            if e.get("category") == "Table" and e.get("bbox")
        ]
        if not table_bboxes_px:
            continue

        raster_rects_pt = _extract_pdf_raster_bboxes(pdf_source, page_index)
        if not raster_rects_pt:
            continue

        existing_pic_rects_px = _existing_picture_rects_px(entries)
        recovered = 0
        for rect_pt in raster_rects_pt:
            rect_px = _rect_pt_to_px(rect_pt, zoom)
            if not _is_inside_any_table(rect_px, table_bboxes_px):
                continue
            if _is_duplicate_of_existing(rect_px, existing_pic_rects_px):
                continue
            # Clamp to image bounds so .crop never goes out of frame.
            x1 = max(0, min(rect_px[0], original_img.width - 1))
            y1 = max(0, min(rect_px[1], original_img.height - 1))
            x2 = max(x1 + 1, min(rect_px[2], original_img.width))
            y2 = max(y1 + 1, min(rect_px[3], original_img.height))
            crop = original_img.crop((x1, y1, x2, y2))
            new_entry = {
                "bbox": [x1, y1, x2, y2],
                "category": "Picture",
                "text": "",
                "image_obj": crop,
                "source": "recovered",
            }
            insert_at = _next_entry_position(entries, (x1, y1, x2, y2))
            entries.insert(insert_at, new_entry)
            existing_pic_rects_px.append((x1, y1, x2, y2))
            recovered += 1

        if recovered:
            log.info(
                "[recover] page %d: recovered %d missing Picture entr%s from PDF",
                page_index, recovered, "y" if recovered == 1 else "ies",
            )
        page["layout_result"] = entries
    return pages
