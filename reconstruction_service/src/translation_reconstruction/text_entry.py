"""Plain-text entry rendering (Title, Section-header, Text, List-item,
Caption, etc.) coordinated with text fitting to prevent overlapping.
"""
from __future__ import annotations

import re
from typing import Dict, Optional


def render_text_entry(ctx, entry: Dict) -> None:
    from .geometry import bbox_px_to_emu
    from .ooxml import build_anchored_textbox_xml, build_paragraph_xml, build_run_xml
    from .text_fit import get_font, measure_width_px, fit_multiline

    text = (entry.get("text") or "").strip()
    if not text:
        return
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    category = entry.get("category", "")
    style = dict(entry.get("style") or {})
    # Left-align everything (no centering). With the hugged box width the text
    # reads from the original left margin.
    alignment = None

    # Strip markdown decorations the VLM may have emitted.
    text = re.sub(r"^\s*#+\s*", "", text)
    text = text.replace("**", "")

    x1, y1, x2, y2 = [float(v) for v in bbox]
    zoom = ctx.zoom
    # Add 6 pt of bottom padding in zoom-space so the OOXML textbox has room
    # for Word's internal line-height rounding that PIL measurement can't capture.
    _PAD_PX = 6.0 * zoom
    padded_y2 = min(y2 + _PAD_PX, ctx.page_h_pt * zoom)
    box_w_pt = max(1.0, (x2 - x1) / zoom)
    box_h_pt = max(1.0, (padded_y2 - y1) / zoom)
    
    base_size_pt = float(style.get("size") or 11.0)
    bold_render = bool(style.get("bold")) or category in (
        "Title", "Section-header", "Page-header", "Page-footer", "Caption",
    )

    # Use the robust fitting engine to calculate font size and line splits
    # maintaining a strict readable minimum floor (6.0 pt) to avoid unreadable text.
    fit_size_pt, lines = fit_multiline(
        text, 
        box_w_pt, 
        box_h_pt, 
        max_size_pt=base_size_pt, 
        min_size_pt=6.0,  # Human-readable font floor
        bold=bold_render
    )
    style["size"] = fit_size_pt

    # Reconstruct text with hard line breaks matching our measurement safely
    if lines:
        processed_text = "\n".join(lines)
    else:
        processed_text = text

    x, y, w, h = bbox_px_to_emu(
        [x1, y1, x2, padded_y2], zoom, ctx.page_w_pt, ctx.page_h_pt,
    )

    runs_xml = build_run_xml(processed_text, style)
    para_xml = build_paragraph_xml(runs_xml, alignment=alignment, line_pt=None)
    
    ctx.xml_chunks.append(
        build_anchored_textbox_xml(
            x, y, w, h, para_xml, ctx._next_id(),
            body_auto_fit=False,  # Fixed boundaries prevent Word from blowing up layouts blindly
        )
    )