"""Geometry plumbing: EMU/pt constants, bbox conversion, page normalisation,
and per-page section setup. Pure helpers — no OOXML strings here."""
from __future__ import annotations

from typing import Dict

from docx import Document
from docx.shared import Emu


EMU_PER_PT = 12700        # OOXML uses English Metric Units: 1 pt = 12700 EMU
EMU_PER_INCH = 914400
DEFAULT_PAGE_WIDTH_PT = 612.0   # US Letter
DEFAULT_PAGE_HEIGHT_PT = 792.0


def normalize_page(page_or_entries) -> Dict:
    """Accept the new {entries, page_width_pt, ...} shape OR a raw list of
    entries (legacy / image fallback). Returns a normalized dict."""
    if isinstance(page_or_entries, dict):
        entries = page_or_entries.get("entries")
        if entries is None and "layout_result" in page_or_entries:
            entries = page_or_entries.get("layout_result")
        return {
            "entries": entries or [],
            "page_width_pt": page_or_entries.get("page_width_pt") or DEFAULT_PAGE_WIDTH_PT,
            "page_height_pt": page_or_entries.get("page_height_pt") or DEFAULT_PAGE_HEIGHT_PT,
            "zoom": page_or_entries.get("zoom") or 1.0,
            "page_index": page_or_entries.get("page_index", 0),
            # carry image_obj through if upstream attached it (Picture render)
            "_raw": page_or_entries,
        }
    if isinstance(page_or_entries, list):
        return {
            "entries": page_or_entries,
            "page_width_pt": DEFAULT_PAGE_WIDTH_PT,
            "page_height_pt": DEFAULT_PAGE_HEIGHT_PT,
            "zoom": 1.0,
            "page_index": 0,
            "_raw": {},
        }
    return {
        "entries": [],
        "page_width_pt": DEFAULT_PAGE_WIDTH_PT,
        "page_height_pt": DEFAULT_PAGE_HEIGHT_PT,
        "zoom": 1.0,
        "page_index": 0,
        "_raw": {},
    }


def add_section_for_page(doc: Document, width_pt: float, height_pt: float,
                         first: bool):
    """Add (or configure) a section sized to the original page with zero
    margins. The first page reuses the default section; subsequent pages
    add a new one preceded by a page break."""
    if first:
        section = doc.sections[0]
    else:
        # New section starts on a new page automatically.
        section = doc.add_section()
    section.page_width = Emu(int(round(width_pt * EMU_PER_PT)))
    section.page_height = Emu(int(round(height_pt * EMU_PER_PT)))
    section.left_margin = Emu(0)
    section.right_margin = Emu(0)
    section.top_margin = Emu(0)
    section.bottom_margin = Emu(0)
    section.header_distance = Emu(0)
    section.footer_distance = Emu(0)
    section.gutter = Emu(0)
    # Each section gets its own header/footer so per-page text routed there
    # doesn't bleed into other pages.
    section.different_first_page_header_footer = False
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    return section


def alignment_for_bbox(bbox, page_w_pt: float, zoom: float) -> str:
    """Pick left/center/right based on where the bbox sits horizontally."""
    if not bbox or len(bbox) != 4:
        return "left"
    x1, _y1, x2, _y2 = bbox
    cx_pt = ((x1 + x2) / 2.0) / max(1e-6, zoom)
    third = page_w_pt / 3.0
    if cx_pt < third:
        return "left"
    if cx_pt > 2 * third:
        return "right"
    return "center"


def bbox_px_to_emu(bbox_px, zoom: float, page_w_pt: float, page_h_pt: float):
    """Convert a 2x-pixel bbox to EMU position+extent, clamped to the page."""
    x1, y1, x2, y2 = bbox_px
    x_pt = max(0.0, x1 / zoom)
    y_pt = max(0.0, y1 / zoom)
    w_pt = max(1.0, (x2 - x1) / zoom)
    h_pt = max(1.0, (y2 - y1) / zoom)
    # Clamp so shapes don't run off the page (Word may render off-page,
    # but viewers handle on-page geometry better).
    if x_pt > page_w_pt:
        x_pt = max(0.0, page_w_pt - 1.0)
    if y_pt > page_h_pt:
        y_pt = max(0.0, page_h_pt - 1.0)
    if x_pt + w_pt > page_w_pt:
        w_pt = max(1.0, page_w_pt - x_pt)
    if y_pt + h_pt > page_h_pt:
        h_pt = max(1.0, page_h_pt - y_pt)
    return (
        int(round(x_pt * EMU_PER_PT)),
        int(round(y_pt * EMU_PER_PT)),
        int(round(w_pt * EMU_PER_PT)),
        int(round(h_pt * EMU_PER_PT)),
    )
