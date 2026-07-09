"""Plain-text entry rendering (Title, Section-header, Text, List-item,
Caption, etc.) coordinated with text fitting to prevent overlapping.
"""
from __future__ import annotations

import re
from typing import Dict, Optional


# Short-label categories that should GROW WIDTH (into the free right margin)
# rather than wrap to a new line and grow height. A translated caption/header is
# usually longer than its source bbox (e.g. Portuguese vs Chinese); wrapping it
# grows the box downward and overlaps the content below. These read as a single
# label line, so widening the box to fit is both truer to the source and avoids
# the overlap.
_WIDTH_GROW_CATEGORIES = {
    "Caption", "Section-Header", "Section-header", "Title",
    "Page-Header", "Page-header",
}
# Leave this much of the page as a right-hand margin when growing width (pt).
_RIGHT_MARGIN_PT = 30.0


def render_text_entry(ctx, entry: Dict) -> None:
    from .geometry import bbox_px_to_emu, EMU_PER_PT
    from .ooxml import build_anchored_textbox_xml, build_paragraph_xml, build_run_xml
    from .text_fit import (
        get_font, has_cjk, wrap_to_width, fit_multiline, measure_width_px,
    )

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

    # Width-grow for short labels: if the text is a single logical line that
    # doesn't fit the source bbox width at its base size, widen the box to the
    # right (up to the page's right margin) so it fits WITHOUT wrapping — instead
    # of wrapping and growing height (which overlaps the content below). Only for
    # single-line labels; multi-line prose keeps its bbox and wraps as before.
    if category in _WIDTH_GROW_CATEGORIES and "\n" not in text:
        _base_pt = float(style.get("size") or 11.0)
        _want_bold = bool(style.get("bold")) or category in (
            "Title", "Section-header", "Section-Header", "Page-header",
            "Page-Header", "Caption",
        )
        _font = get_font(max(1, int(round(_base_pt))), bold=_want_bold)
        if _font is not None:
            _cur_w_pt = max(1.0, (x2 - x1) / zoom)
            _x1_pt = x1 / zoom
            # One-line width the text needs at base size (+ a little slack).
            _text_w_pt = measure_width_px(
                text, _font, int(round(_base_pt))
            ) + 6.0
            # Room available from the box's left edge to the page right margin.
            _avail_w_pt = max(0.0, (ctx.page_w_pt - _RIGHT_MARGIN_PT) - _x1_pt)
            if _text_w_pt > _cur_w_pt and _avail_w_pt > _cur_w_pt:
                # Grow to hold the whole text on one line when it fits the
                # available room; otherwise take ALL the available width so it
                # wraps into as FEW lines as possible (never stay narrow — that
                # is what forced the extra wrapped lines + height overflow).
                _new_w_pt = min(_text_w_pt, _avail_w_pt)
                x2 = x1 + _new_w_pt * zoom

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

    x, y, w, h = bbox_px_to_emu(
        [x1, y1, x2, padded_y2], zoom, ctx.page_w_pt, ctx.page_h_pt,
    )

    # `lines is None` means the text can't fit the OCR bbox even at the 6 pt
    # floor. Instead of clipping it to an ellipsis (silently losing content —
    # e.g. long footers/titles the OCR gave a too-short bbox), keep the floor
    # size, wrap the FULL text, and GROW the box downward (spAutoFit) so every
    # line is visible. Fixed boundaries are still used for text that fits.
    body_auto_fit = lines is None
    if lines is None:
        floor_pt = 6.0
        style["size"] = min(base_size_pt, floor_pt)
        font = get_font(max(1, int(round(floor_pt))), bold=bold_render)
        if font is not None:
            wrapped = wrap_to_width(text, font, box_w_pt * 0.93, int(round(floor_pt)))
            processed_text = "\n".join(wrapped)
            asc, desc = font.getmetrics()
            natural_h = asc + desc
            if has_cjk(text):
                natural_h = max(natural_h, floor_pt * 1.2)
            line_h_pt = natural_h * 1.10
            needed_h_pt = max(box_h_pt, line_h_pt * len(wrapped))
            h = max(h, int(round(needed_h_pt * EMU_PER_PT)))
        else:
            processed_text = text
    else:
        style["size"] = fit_size_pt
        processed_text = "\n".join(lines) if lines else text

    runs_xml = build_run_xml(processed_text, style)
    para_xml = build_paragraph_xml(runs_xml, alignment=alignment, line_pt=None)

    ctx.xml_chunks.append(
        build_anchored_textbox_xml(
            x, y, w, h, para_xml, ctx._next_id(),
            body_auto_fit=body_auto_fit,
        )
    )