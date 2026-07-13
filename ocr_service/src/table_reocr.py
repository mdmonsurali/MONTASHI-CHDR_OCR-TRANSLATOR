"""Recover table rows Chandra dropped on the full-page OCR pass by re-OCRing
the cropped table region.

On a dense full page Chandra sometimes drops an entire table ``<tr>`` — its text
AND its ``<img>`` placeholder — collapsing a product group's rowspan (e.g. a
3-row group emitted as ``rowspan=2`` with one image instead of two). The missing
row never reaches reconstruction, so the table renders short a row and the
recovered diagrams (``picture_recovery`` finds every embedded PDF raster) then
outnumber the ``<img>`` cells and pile into one merged cell.

The model transcribes the same table correctly when given ONLY the cropped table
region (a much smaller image, no competing full-page content). This module
detects the drop — embedded rasters inside the table bbox exceed the table's
``<img>`` cells — re-OCRs the crop, and swaps in the fuller HTML, but ONLY when
the result is strictly better so a bad re-OCR can never regress a good table.

Native-PDF only: the detection and crop both need the embedded-raster geometry
and the rendered page raster. Scanned pages / image uploads have no embedded
raster to count and are handled by ``cell_picture_recovery`` instead.

Public surface
--------------
recover_dropped_table_rows(pages) — async; mutates each page's Table entries in
place, replacing ``text`` with a re-OCR'd table when a dropped row is detected
and the re-OCR is an improvement. Returns the same ``pages`` list.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from PIL import Image

from chandra_ocr import ocr_image_async
from picture_recovery import (
    _bbox_contains,
    _extract_pdf_raster_bboxes,
    _rect_pt_to_px,
)
# Same grid parser the renderer uses, so the <img>-cell count matches exactly
# what reconstruction will build.
from ocr_reconstruction.table import parse_html_table_rows, parse_table_grid

log = logging.getLogger("ocr_service")

# Margin (px) around the table bbox when cropping — a couple of the source
# tables sit flush against a ruling the model reads better with a little air.
_CROP_MARGIN_PX = 12

_TR_RE = re.compile(r"<tr\b", re.IGNORECASE)


def _img_cell_count(table_html: str) -> Tuple[int, int]:
    """(<img>-cell count, data-row count) for a table's HTML, indexed on the
    SAME grid the renderer builds. Data-row count excludes the header row."""
    rows = parse_html_table_rows(table_html)
    if not rows:
        return 0, 0
    _mc, n_rows, _ca, _occ, img_by_anchor = parse_table_grid(rows)
    img_cells = sum(1 for v in img_by_anchor.values() if v > 0)
    data_rows = sum(0 if is_header else 1 for _cells, is_header in rows)
    return img_cells, data_rows


def _rasters_inside_table(
    pdf_source: str, page_index: int, zoom: float,
    table_bbox_px: Tuple[int, int, int, int],
) -> int:
    """Count embedded PDF rasters whose on-page rect falls inside the table
    bbox (page-image pixels). This is the number of real diagrams the table
    holds, independent of what the OCR HTML claims."""
    rasters = _extract_pdf_raster_bboxes(pdf_source, page_index)
    n = 0
    for rect_pt in rasters:
        if _bbox_contains(table_bbox_px, _rect_pt_to_px(rect_pt, zoom)):
            n += 1
    return n


def _extract_table_html(entries: List[dict]) -> Optional[str]:
    """The ``text`` of the single Table entry a table-crop re-OCR should return.
    Prefers the largest table by row count when the model emits more than one
    (it should emit exactly one for a table crop)."""
    tables = [e for e in entries if e.get("category") == "Table" and e.get("text")]
    if not tables:
        return None
    return max(tables, key=lambda e: len(_TR_RE.findall(e["text"]))).get("text")


async def _reocr_one_table(
    table_entry: dict, page_img: Image.Image,
    pdf_source: str, page_index: int, zoom: float,
) -> bool:
    """Detect a dropped row in `table_entry` and, if found, re-OCR its crop and
    replace `text` with the fuller HTML. Returns True when the table was
    replaced. Never raises on a single bad table — logs and returns False."""
    html = table_entry.get("text") or ""
    bbox = table_entry.get("bbox")
    if not html or not bbox or len(bbox) != 4:
        return False
    tbl = tuple(int(v) for v in bbox)
    x1, y1, x2, y2 = tbl
    if x2 <= x1 or y2 <= y1:
        return False

    img_cells, data_rows = _img_cell_count(html)
    rasters = _rasters_inside_table(pdf_source, page_index, zoom, tbl)
    # Trigger only when the table demonstrably lost an image (and thus, on these
    # docs, its row): more real diagrams than the grid has image cells.
    if rasters <= img_cells:
        return False

    log.info(
        "[table-reocr] page %d: table has %d embedded raster(s) but %d <img> "
        "cell(s) — re-OCRing the table crop to recover the dropped row(s)",
        page_index, rasters, img_cells,
    )

    # Crop the table region (+ margin) and re-OCR just that.
    cx0 = max(0, x1 - _CROP_MARGIN_PX)
    cy0 = max(0, y1 - _CROP_MARGIN_PX)
    cx1 = min(page_img.width, x2 + _CROP_MARGIN_PX)
    cy1 = min(page_img.height, y2 + _CROP_MARGIN_PX)
    if cx1 <= cx0 or cy1 <= cy0:
        return False
    crop = page_img.crop((cx0, cy0, cx1, cy1)).convert("RGB")

    try:
        entries = await ocr_image_async(crop)
    except Exception as exc:
        log.warning("[table-reocr] page %d: re-OCR failed: %s", page_index, exc)
        return False

    new_html = _extract_table_html(entries)
    if not new_html:
        log.info("[table-reocr] page %d: re-OCR returned no table; keeping "
                 "original", page_index)
        return False

    new_img_cells, new_data_rows = _img_cell_count(new_html)
    # Acceptance guard — replace ONLY when strictly better and not overshooting:
    #   * the re-OCR is a real table (parses to >=1 data row),
    #   * it did not LOSE rows,
    #   * it recovered image cell(s) but no more than the true raster count.
    if (
        new_data_rows >= 1
        and new_data_rows >= data_rows
        and new_img_cells > img_cells
        and new_img_cells <= rasters
    ):
        table_entry["text"] = new_html
        table_entry["source"] = "row-reocr"
        log.info(
            "[table-reocr] page %d: replaced table (rows %d→%d, <img> %d→%d)",
            page_index, data_rows, new_data_rows, img_cells, new_img_cells,
        )
        return True

    log.info(
        "[table-reocr] page %d: re-OCR not an improvement (rows %d→%d, <img> "
        "%d→%d, rasters %d); keeping original",
        page_index, data_rows, new_data_rows, img_cells, new_img_cells, rasters,
    )
    return False


async def recover_dropped_table_rows(pages: List[dict]) -> List[dict]:
    """For every native-PDF page, re-OCR any table whose embedded-raster count
    exceeds its ``<img>``-cell count (Chandra dropped a row+image) and swap in
    the fuller HTML when the re-OCR is an improvement. Mutates `pages` in place.

    Runs BEFORE ``recover_missing_pictures`` so picture recovery and the
    reconstruction-side cell assignment operate on the corrected grid. A no-op
    for pages with no ``pdf_source``/raster and for tables whose counts already
    agree, so well-formed documents are untouched (and cost no extra OCR calls).
    """
    for page in pages:
        pdf_source = page.get("pdf_source")
        page_img: Optional[Image.Image] = (
            page.get("original_image") or page.get("image")
        )
        if not pdf_source or page_img is None:
            continue
        entries: List[dict] = page.get("layout_result") or []
        tables = [e for e in entries if e.get("category") == "Table"]
        if not tables:
            continue
        page_index = int(page.get("page_index", 0))
        zoom = float(page.get("zoom") or 1.0)

        for table in tables:
            await _reocr_one_table(table, page_img, pdf_source, page_index, zoom)

    return pages
