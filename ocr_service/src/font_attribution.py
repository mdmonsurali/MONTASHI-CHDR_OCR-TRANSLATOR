"""Attach per-entry font style to OCR layout entries.

For native PDFs (and DOCX-converted-to-PDF) we read font/size/bold/italic/
color directly from the PDF text layer via PyMuPDF spans, then assign each
layout entry the dominant style of the spans that fall inside its bbox.

For raster inputs (JPG/PNG) or scanned PDFs with no text layer, we fall
back to a category-driven heuristic.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import math
import re


HEURISTIC_BOLD_CATEGORIES = {
    "Title", "Section-header", "Page-header", "Page-footer", "Caption",
}
HEURISTIC_ITALIC_CATEGORIES = {"Caption", "Footnote"}
DEFAULT_FONT = "Calibri"
DEFAULT_SIZE_PT = 11.0


# East-Asian Wide / Fullwidth codepoints: Han ideographs, kana, Hangul, CJK
# punctuation and fullwidth forms. These advance ~1 em (double a Latin glyph)
# in essentially every font, which is what makes them relevant to line-count
# estimation. Kept in sync (by intent) with text_fit._CJK_RE.
_WIDE_CH_RE = re.compile(
    r"[ᄀ-ᇿ⺀-⻿　-〿぀-ヿ㄰-㆏"
    r"㐀-䶿一-鿿ꀀ-꓏가-힯豈-﫿"
    r"︰-﹏＀-￯]"
)


def _is_wide_char(ch: str) -> bool:
    return bool(_WIDE_CH_RE.match(ch))


def _bbox_px_to_pt(bbox_px, zoom: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_px
    return (x1 / zoom, y1 / zoom, x2 / zoom, y2 / zoom)


def _span_center(span_bbox_pt: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = span_bbox_pt
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _point_in_box(px: float, py: float, box: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= px <= x1 and y0 <= py <= y1


def _estimate_line_count(entry: Dict, zoom: float) -> int:
    """Estimate how many lines of text live inside the entry's bbox.

    Strategy: divide the rendered text width (≈ len(text) × glyph_advance)
    by the bbox width to get the wrapped-line count. The advance is
    fictional but cancels out when text-width-to-bbox-width is the same
    proportion as line-height to a 1-line height.

    Falls back to 1 when there's no text or the bbox is degenerate. Caps
    at the number of newlines + 1 in the text (so explicit hard-wraps are
    respected even if the math underestimates).
    """
    bbox = entry.get("bbox") or [0, 0, 0, 0]
    text = (entry.get("text") or "").strip()
    if not text:
        return 1
    width_px = max(1.0, bbox[2] - bbox[0])
    # Estimated rendered text width. The absolute advance is fictional; only
    # the text-width-to-bbox-width RATIO matters. Advance is per-character and
    # SCRIPT-AWARE: a Latin glyph advances ~half an em, but a CJK ideograph /
    # fullwidth form advances a full em (~2x). Using one Latin-calibrated
    # advance for everything under-measures Chinese text by ~2x, collapses the
    # wrapped-line count to 1, and then `_heuristic_style` divides the whole
    # multi-line bbox height by a single line — inflating the font size by the
    # true line count. Counting wide glyphs at 2x advance fixes that at the
    # root and stays correct for pure-Latin, pure-CJK, and mixed text.
    narrow_advance = 7.0 * max(zoom, 1.0)
    wide_advance = 2.0 * narrow_advance
    estimated_text_width = sum(
        wide_advance if _is_wide_char(ch) else narrow_advance for ch in text
    )
    # Wrapped-line count is a CEILING, not a round: text that spans 1.38 line
    # widths wraps to 2 lines, not 1. `round()` under-counted every paragraph
    # whose overflow fraction was below 0.5 — collapsing it toward a single
    # line and inflating the derived font size. The 0.25 tolerance absorbs the
    # measurement slop so a paragraph that just barely exceeds an exact line
    # count (e.g. 2.05) isn't pushed to the next line.
    ratio = estimated_text_width / width_px
    by_width = max(1, math.ceil(ratio - 0.25))
    by_newline = text.count("\n") + 1
    return max(by_newline, min(by_width, 50))  # cap at 50 — sanity guard


def _heuristic_style(entry: Dict, zoom: float) -> Dict:
    bbox = entry.get("bbox") or [0, 0, 0, 0]
    height_px = max(0.0, bbox[3] - bbox[1])
    lines = _estimate_line_count(entry, zoom)
    # bbox is in pixels → height in points ≈ height_px / zoom.
    # Divide by line count so multi-line paragraphs don't report giant sizes.
    # Real glyph height is typically ~70% of line-box height (descenders + padding).
    per_line_height_px = height_px / max(lines, 1)
    size_pt = (per_line_height_px / max(zoom, 1e-6)) * 0.7
    if size_pt < 6.0 or size_pt > 96.0:
        size_pt = DEFAULT_SIZE_PT

    category = entry.get("category", "")
    return {
        "font": DEFAULT_FONT,
        "size": round(size_pt, 1),
        "bold": category in HEURISTIC_BOLD_CATEGORIES,
        "italic": category in HEURISTIC_ITALIC_CATEGORIES,
        "color": [0, 0, 0],
        "source": "heuristic",
    }


def _dominant_style(matched_spans: List[Dict]) -> Optional[Dict]:
    """Pick the style covering the most text by character count."""
    if not matched_spans:
        return None

    # Bucket by (font, rounded size, bold, italic, color); pick the heaviest.
    weights: Dict[Tuple, int] = defaultdict(int)
    representatives: Dict[Tuple, Dict] = {}
    for span in matched_spans:
        key = (
            span.get("font", "") or DEFAULT_FONT,
            round(float(span.get("size") or DEFAULT_SIZE_PT), 1),
            bool(span.get("bold")),
            bool(span.get("italic")),
            tuple(span.get("color_rgb") or (0, 0, 0)),
        )
        weights[key] += len((span.get("text") or "").strip())
        representatives.setdefault(key, span)

    if not weights:
        return None

    winner_key = max(weights.items(), key=lambda kv: kv[1])[0]
    font, size, bold, italic, color = winner_key
    return {
        "font": font or DEFAULT_FONT,
        "size": size,
        "bold": bold,
        "italic": italic,
        "color": list(color),
        "source": "pdf",
    }


def attribute_page(
    entries: List[Dict],
    spans: List[Dict],
    zoom: float,
) -> List[Dict]:
    """Mutate each entry by adding a 'style' dict. Returns the same list."""
    for entry in entries:
        bbox_px = entry.get("bbox")
        category = entry.get("category", "")
        if not bbox_px or len(bbox_px) != 4:
            entry["style"] = _heuristic_style(entry, zoom)
            continue
        if category == "Picture":
            # Pictures don't carry text; skip style attribution.
            continue

        bbox_pt = _bbox_px_to_pt(bbox_px, zoom)
        matched = [
            s for s in spans
            if _point_in_box(*_span_center(s["bbox_pt"]), bbox_pt)
        ]
        style = _dominant_style(matched) or _heuristic_style(entry, zoom)
        entry["style"] = style
    return entries