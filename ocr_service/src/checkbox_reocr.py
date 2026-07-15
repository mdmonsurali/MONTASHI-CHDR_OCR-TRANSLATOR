"""Recover form checkboxes Chandra dropped on the full-page OCR pass by
re-OCRing the cropped block region.

Empty/unchecked checkboxes are small, faint, low-contrast marks — the first
thing lost when a dense full page is downscaled to the model's input size. On
such a page Chandra sometimes transcribes an option line as its label text
alone (``是, 否`` / ``Yes No``) with no ``<input>`` box at all, so the boxes
never reach reconstruction and the rendered form shows bare labels.

The model reads those same boxes when handed ONLY the tight block crop (far
more pixels per checkbox, no competing full-page content). This module detects
the likely drop — a Text / List-Group / Form block whose text has option-label
patterns but ZERO checkbox glyphs or ``<input>`` tags — re-OCRs the block crop,
and swaps in the recovered text, but ONLY when the re-OCR strictly adds boxes
without losing the original wording, so a bad re-OCR can never regress a good
block.

Works on BOTH native PDFs and scanned pages / image uploads: detection and crop
use only the block bbox and the rendered page raster (``original_image``), which
every page carries — unlike ``table_reocr`` (native-only, needs embedded-raster
geometry).

Public surface
--------------
recover_dropped_checkboxes(pages) — async; mutates each page's candidate entries
in place, replacing ``text`` with re-OCR'd text when a dropped box is detected
and the re-OCR is an improvement. Returns the same ``pages`` list.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from PIL import Image

from chandra_ocr import ocr_image_async

log = logging.getLogger("ocr_service")

# Categories that carry standalone form options (never Table — those go through
# table_reocr / the table renderer's own <input> handling).
_CANDIDATE_CATEGORIES = {"Text", "List-Group", "Form", "Caption"}

# Any checkbox / radio / tick glyph, or a literal <input> tag: presence of ANY
# of these means the block already has its boxes, so we skip it.
_HAS_BOX_RE = re.compile(r"<input\b|[☐-☒○●◯✓✔✗✘]", re.IGNORECASE)

# Option-label patterns that strongly imply a checkbox/radio line whose boxes
# may have been dropped. Chinese yes/no/known forms plus Latin equivalents, and
# the generic "label: A B" / "label: A, B" two-option shape after a colon.
_OPTION_HINT_RES = (
    # 是/否/未知, 有/无, 正常/异常, 合格/不合格 pairs (Chinese form staples)
    re.compile(r"[:：]\s*是\b.*?\b否"),
    re.compile(r"[:：]\s*有\b.*?\b无"),
    re.compile(r"正常\b.*?\b(?:异常|非正常)"),
    re.compile(r"合格\b.*?\b不合格"),
    # Latin yes/no / pass-fail after a colon.
    re.compile(r"[:：]\s*(?:yes|pass|normal)\b.*?\b(?:no|fail|abnormal)\b",
               re.IGNORECASE),
)

# Margin (px) around the block bbox when cropping — a little air helps the model
# resolve boxes sitting flush against ruling or text.
_CROP_MARGIN_PX = 14


def _looks_like_dropped_checkbox_block(text: str) -> bool:
    """True when `text` has option-label patterns but no box glyph/<input>,
    i.e. Chandra probably dropped the checkboxes for this block."""
    if not text or _HAS_BOX_RE.search(text):
        return False
    return any(r.search(text) for r in _OPTION_HINT_RES)


def _box_count(text: str) -> int:
    """Number of checkbox/radio glyphs plus <input> tags in `text`."""
    glyphs = len(re.findall(r"[☐-☒○●◯✓✔✗✘]", text or ""))
    inputs = len(re.findall(r"<input\b", text or "", re.IGNORECASE))
    return glyphs + inputs


def _alnum_cjk(text: str) -> str:
    """Content signature: letters/digits/CJK only, lowercased. Used to check the
    re-OCR kept the original wording (ignoring boxes/whitespace/punctuation)."""
    return re.sub(r"[^0-9a-z㐀-䶿一-鿿]", "",
                  (text or "").lower())


def _recovered_text_for_crop(entries: List[dict]) -> Optional[str]:
    """Concatenated text of the re-OCR'd crop's blocks, in reading order.
    A single option line may come back as one or a few small blocks."""
    parts = [(e.get("text") or "").strip()
             for e in entries
             if e.get("category") != "Table" and (e.get("text") or "").strip()]
    if not parts:
        return None
    return "\n".join(parts)


async def _reocr_one_block(entry: dict, page_img: Image.Image,
                           page_index: int) -> bool:
    """Detect dropped checkboxes in `entry` and, if found, re-OCR its crop and
    replace `text` with the recovered wording. Returns True when replaced. Never
    raises on a single bad block — logs and returns False."""
    text = entry.get("text") or ""
    bbox = entry.get("bbox")
    if not _looks_like_dropped_checkbox_block(text):
        return False
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return False

    cx0 = max(0, x1 - _CROP_MARGIN_PX)
    cy0 = max(0, y1 - _CROP_MARGIN_PX)
    cx1 = min(page_img.width, x2 + _CROP_MARGIN_PX)
    cy1 = min(page_img.height, y2 + _CROP_MARGIN_PX)
    if cx1 <= cx0 or cy1 <= cy0:
        return False

    log.info(
        "[checkbox-reocr] page %d: block looks like it lost its checkboxes "
        "(%r) — re-OCRing the block crop", page_index, text[:60],
    )
    crop = page_img.crop((cx0, cy0, cx1, cy1)).convert("RGB")
    try:
        entries = await ocr_image_async(crop)
    except Exception as exc:
        log.warning("[checkbox-reocr] page %d: re-OCR failed: %s",
                    page_index, exc)
        return False

    new_text = _recovered_text_for_crop(entries)
    if not new_text:
        return False

    # Acceptance guard — replace ONLY when the re-OCR:
    #   * actually recovered box(es) the original lacked, AND
    #   * preserved the original wording (content signature is a superset), so
    #     a garbled or truncated re-OCR can never overwrite good text.
    new_boxes = _box_count(new_text)
    if new_boxes == 0:
        log.info("[checkbox-reocr] page %d: re-OCR still found no boxes; "
                 "keeping original", page_index)
        return False
    orig_sig = _alnum_cjk(text)
    new_sig = _alnum_cjk(new_text)
    if orig_sig and orig_sig not in new_sig:
        log.info("[checkbox-reocr] page %d: re-OCR changed the wording "
                 "(sig mismatch); keeping original to avoid regression",
                 page_index)
        return False

    entry["text"] = new_text
    entry["source"] = "checkbox-reocr"
    log.info("[checkbox-reocr] page %d: recovered %d checkbox(es) → %r",
             page_index, new_boxes, new_text[:80])
    return True


async def recover_dropped_checkboxes(pages: List[dict]) -> List[dict]:
    """For every page, re-OCR any standalone form block whose text has option
    labels but no checkbox glyph/<input> (Chandra dropped the boxes) and swap in
    the recovered wording when the re-OCR strictly adds boxes without losing the
    original text. Mutates `pages` in place.

    A no-op for blocks that already carry their boxes and for blocks with no
    option-label pattern, so well-formed documents are untouched (and cost no
    extra OCR calls). Table blocks are skipped — they go through `table_reocr`
    and the table renderer's own <input> handling.
    """
    for page in pages:
        page_img: Optional[Image.Image] = (
            page.get("original_image") or page.get("image")
        )
        if page_img is None:
            continue
        entries: List[dict] = page.get("layout_result") or []
        page_index = int(page.get("page_index", 0))
        for entry in entries:
            if entry.get("category") not in _CANDIDATE_CATEGORIES:
                continue
            await _reocr_one_block(entry, page_img, page_index)
    return pages
