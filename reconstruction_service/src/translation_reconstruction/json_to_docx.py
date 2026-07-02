"""Top-level entry: turn the OCR layout JSON into a DOCX whose page count,
page size, and per-entry positions match the source document. Delegates
each entry kind to its dedicated renderer module.
"""
from __future__ import annotations

from io import BytesIO
from typing import Dict, List

from docx import Document
from docx.shared import Pt, RGBColor

from .geometry import (
    add_section_for_page, normalize_page,
)
from .picture import render_standalone_picture
from .table import (
    render_table, parse_html_table_rows, parse_markdown_table,
    parse_table_grid, compute_col_weights,
)
from .formula import render_formula
from .text_entry import render_text_entry
from .shape_context import ShapeContext
from .reflow import layout_page


# ─── Style normalisation ────────────────────────────────────────────────
#
# After translation the styles in the layout JSON are still the source-PDF's
# styles — CJK CIDFont names that downstream Word will substitute with
# something arbitrary, and per-entry sizes that drift across pages (a
# Page-header rendered at 7pt on one page and 9.8pt on another). The
# translator service can't fix this without re-fitting every box, so we do
# it here right before rendering.
#
# Two goals:
#   1. ALL entries use one universally-available font ("Calibri") so the
#      document renders consistently regardless of where it's opened. The
#      CIDFont substitution behaviour was the main reason text appeared
#      clipped — substituted fonts almost always have different metrics
#      than the original.
#   2. Each category gets ONE base size, so siblings of the same role
#      (every Section-header, every Caption, every Page-footer) read
#      identically. The per-entry text_fit step still shrinks below this
#      base when necessary to avoid clipping the actual rendered text, but
#      it starts from a known sensible value, not a random source size.
#
# The base sizes mirror Word's defaults for an A4 report (Title 14, body
# 10, caption/footer 9). Page-header is intentionally a touch smaller than
# body so a running-head doesn't fight the body for attention.

UNIFIED_FONT = "Calibri"

CATEGORY_BASE_SIZE_PT: Dict[str, float] = {
    "Title": 14.0,
    "Section-header": 12.0,
    "Page-header": 9.0,
    "Page-footer": 9.0,
    "Caption": 10.0,
    "Footnote": 8.5,
    "List-item": 10.5,
    "Text": 10.5,
    "Table": 9.5,
    "Formula": 11.0,
    "Picture": 11.0,  # unused — pictures have no text
}

# Categories that should ALWAYS render in bold weight, regardless of what
# the source style said. Knowing this up-front matters because the text-fit
# step measures with the bold metric variant — bold glyphs are 12-15% wider
# than regular, and using the wrong weight produced "fitting" sizes that
# Word then re-wrapped onto a clipped second line.
ALWAYS_BOLD_CATEGORIES = {
    "Title", "Section-header", "Page-header", "Page-footer", "Caption",
}

DEFAULT_BASE_SIZE_PT = 10.5


def _normalize_entry_styles(layout_results) -> None:
    """Rewrite every entry's `style` so font, size, and weight are
    predictable.

    Mutates `layout_results` in place. Each entry's style.font becomes
    `UNIFIED_FONT`; each entry's style.size becomes the category's base
    size; headings/headers/footers/captions get style.bold=True. Italic
    and color flags pass through untouched so translated emphasis (and
    deliberately non-black runs) survive.
    """
    for raw_page in layout_results:
        if isinstance(raw_page, dict):
            entries = (
                raw_page.get("entries")
                or raw_page.get("layout_result")
                or []
            )
        elif isinstance(raw_page, list):
            entries = raw_page
        else:
            continue
        for entry in entries:
            cat = entry.get("category") or ""
            style = dict(entry.get("style") or {})
            style["font"] = UNIFIED_FONT
            style["size"] = CATEGORY_BASE_SIZE_PT.get(cat, DEFAULT_BASE_SIZE_PT)
            if cat in ALWAYS_BOLD_CATEGORIES:
                style["bold"] = True
            entry["style"] = style


def _parse_table_entry_rows(entry: Dict):
    """Parse the table HTML on `entry` into rows. None when not a table or
    when parsing yields nothing."""
    text = (entry.get("text") or "").strip()
    if not text:
        return None
    if "<table" in text.lower():
        return parse_html_table_rows(text)
    md_rows = parse_markdown_table(text)
    if md_rows:
        return [([(c, 1, 1) for c in r], False) for r in md_rows]
    return None


def _link_table_continuations(layout_results) -> None:
    """Detect tables that continue from a previous page and propagate the
    head table's column-weight vector onto them so column widths stay
    consistent across the page break.

    Heuristic: a Table entry is a continuation of the most recent prior
    Table entry when it has no `<thead>`, the parsed grid has the same
    `max_cols`, and it sits near the top of its page (just below the page
    header). The chain detector also re-computes the shared weights using
    cells from EVERY page in the chain — so a column that only carries
    long text on a continuation page still gets enough width on the head
    page.
    """
    chain_parent_entry = None
    chain_max_cols = 0
    chain_anchors: Dict = {}
    chain_row_offset = 0

    def _finalize_chain():
        if chain_parent_entry is None or not chain_anchors:
            return
        weights = compute_col_weights(chain_anchors, chain_max_cols)
        for ent in chain_parent_entry.get("_table_chain", [chain_parent_entry]):
            ent["_shared_col_weights"] = list(weights)

    for raw_page in layout_results:
        if isinstance(raw_page, dict):
            entries = (
                raw_page.get("entries")
                or raw_page.get("layout_result")
                or []
            )
        elif isinstance(raw_page, list):
            entries = raw_page
        else:
            entries = []
        # The page-header bottom — used to decide whether a table is at the
        # top of the page (i.e. immediately after the header band). When
        # there's no detected page-header band, treat anything in the top
        # ~15% of the page as 'at top'.
        header_bottoms = [
            (e.get("bbox") or [0, 0, 0, 0])[3]
            for e in entries
            if e.get("category") == "Page-header"
        ]
        page_header_y2 = max(header_bottoms) if header_bottoms else 0

        # Find the first table on the page (if any) and remember whether any
        # non-table content precedes it — that's the only place a multi-page
        # continuation can sit. A second table on the same page is always a
        # new logical table.
        page_tables = [e for e in entries if e.get("category") == "Table"]
        first_table_id = id(page_tables[0]) if page_tables else None

        for entry in entries:
            if entry.get("category") != "Table":
                continue
            rows = _parse_table_entry_rows(entry)
            if not rows:
                continue
            text = (entry.get("text") or "").strip()
            has_thead = "<thead" in text.lower()
            max_cols, n_rows, cell_anchors, _occ = parse_table_grid(rows)

            bbox = entry.get("bbox") or [0, 0, 0, 0]
            tbl_top = bbox[1] if len(bbox) >= 2 else 0
            at_top_of_page = (tbl_top - page_header_y2) <= 60

            is_continuation = (
                chain_parent_entry is not None
                and not has_thead
                and max_cols == chain_max_cols
                and at_top_of_page
                and id(entry) == first_table_id
            )
            if is_continuation:
                for (r, c), v in cell_anchors.items():
                    chain_anchors[(chain_row_offset + r, c)] = v
                chain_row_offset += n_rows
                chain = chain_parent_entry.setdefault(
                    "_table_chain", [chain_parent_entry]
                )
                chain.append(entry)
            else:
                _finalize_chain()
                chain_parent_entry = entry
                chain_anchors = dict(cell_anchors)
                chain_max_cols = max_cols
                chain_row_offset = n_rows
                chain_parent_entry["_table_chain"] = [chain_parent_entry]

    _finalize_chain()


def json_to_docx(layout_results, output_path="output.docx"):
    """Build a DOCX whose page count, page size, and per-entry positions
    match the source document.

    `layout_results` is a list of page envelopes (new shape) or a list of
    entry-lists (legacy). When called from the live pipeline it's the same
    `pages` list that `process_pictures` returned, which carries
    `image_obj` on Picture entries — those get embedded.
    """
    doc = Document()

    # Keep a sensible Normal style so any inline run we don't override
    # still has a reasonable default. Zero the paragraph spacing too:
    # python-docx's default Normal carries 1.15x line spacing and 10pt
    # space-after, which would otherwise inflate any paragraph we don't
    # explicitly override and reintroduce the vertical-clip bug.
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.font.color.rgb = RGBColor(0, 0, 0)
    pf = style.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0

    shape_counter = 1000  # docPr ids must be unique and >0

    # Normalize every entry's font + base size BEFORE table chaining and
    # rendering. Downstream text-fit (fit_multiline) will still shrink any
    # entry whose translated text doesn't fit at the base size — this just
    # pins a consistent starting point per role.
    _normalize_entry_styles(layout_results)

    # Annotate Table entries that belong to a multi-page chain with a shared
    # column-weight vector so widths stay consistent across the page break.
    _link_table_continuations(layout_results)

    # Reflow every source page in PLACE: one source page -> exactly one
    # physical page. Boxes grow to the height their translated text needs at
    # the uniform font and push column-neighbours below them down (pictures
    # flow too), but nothing moves to another page — the page grows taller to
    # hold its own content, so the source page count is preserved and every
    # entry stays on its original page. A page that would exceed Word's height
    # ceiling is scaled down uniformly (that page only) to avoid clipping.
    physical_pages: List[Dict] = []
    # Map from entry id → original (pre-reflow) bbox for each Table entry.
    # Reflow mutates table bboxes in place (y shifts/grows), but table-contained
    # pictures are excluded from reflow and keep their original coords. We need
    # the original table bboxes to correctly identify which pictures belong inside
    # which table after reflow has run.
    pre_reflow_table_bboxes: Dict[int, List] = {}
    for raw_page in layout_results:
        page = normalize_page(raw_page)
        raw_entries: List[Dict] = []
        if isinstance(raw_page, dict):
            raw_entries = (
                raw_page.get("layout_result")
                or raw_page.get("entries")
                or []
            )
        elif isinstance(raw_page, list):
            raw_entries = raw_page
        entries = page["entries"] or raw_entries
        for e in entries:
            if e.get("category") == "Table" and e.get("bbox"):
                pre_reflow_table_bboxes[id(e)] = list(e["bbox"])
            # Snapshot each picture's pre-reflow bbox. Pictures are frozen
            # during cascade push but may still be scaled by layout_page;
            # the snapshot keeps picture→cell assignment in the original
            # coordinate frame (see render_table / _assign_pictures_to_cells).
            if e.get("category") == "Picture" and e.get("bbox"):
                e["_orig_bbox"] = list(e["bbox"])
        physical_pages.extend(layout_page(
            entries,
            float(page["page_width_pt"]),
            float(page["page_height_pt"]),
            float(page["zoom"]),
        ))

    for idx, page in enumerate(physical_pages):
        entries = page["entries"]
        page_w_pt = float(page["page_width_pt"])
        page_h_pt = float(page["page_height_pt"])
        page_zoom = float(page["zoom"])

        add_section_for_page(
            doc,
            page_w_pt,
            page_h_pt,
            first=(idx == 0),
        )

        # Page-header / Page-footer entries are rendered as positioned
        # floating textboxes inside the body (same path as Text/Title/etc.)
        # so they land at their original bbox y-position. Routing them into
        # Word's section header/footer container made them stack from y=0
        # of the page, which broke multi-line headers like a logo block.

        ctx = ShapeContext(
            doc,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            zoom=page_zoom,
            shape_id_start=shape_counter,
        )

        tables = [e for e in entries if e.get("category") == "Table"]

        def _bbox_inside(inner, outer) -> bool:
            if not inner or not outer or len(inner) != 4 or len(outer) != 4:
                return False
            return (inner[0] >= outer[0] and inner[2] <= outer[2]
                    and inner[1] >= outer[1] and inner[3] <= outer[3])

        def _picture_in_any_table(pic_entry) -> bool:
            # Compare in the PRE-REFLOW frame: pictures now cascade-move, so
            # their live bbox is post-move/scale; the snapshot is the frame
            # the table snapshots were taken in.
            pb = pic_entry.get("_orig_bbox") or pic_entry.get("bbox")
            for t in tables:
                orig_tb = pre_reflow_table_bboxes.get(id(t)) or t.get("bbox")
                if _bbox_inside(pb, orig_tb):
                    return True
            return False

        standalone_pics = [
            e for e in entries
            if e.get("category") == "Picture" and not _picture_in_any_table(e)
        ]
        # Bucket contained pictures by their owning table so each table
        # renders its own photos inside the matching cell.
        pics_per_table: Dict[int, List[Dict]] = {}
        contained_no_image = []
        for e in entries:
            if e.get("category") != "Picture":
                continue
            pb = e.get("_orig_bbox") or e.get("bbox")
            for t in tables:
                orig_tb = pre_reflow_table_bboxes.get(id(t)) or t.get("bbox")
                if _bbox_inside(pb, orig_tb):
                    if e.get("image_obj") is not None:
                        pics_per_table.setdefault(id(t), []).append(e)
                    else:
                        # Fall back to floating anchor on top of the table —
                        # better than disappearing entirely.
                        contained_no_image.append(e)
                    break

        for entry in standalone_pics:
            render_standalone_picture(ctx, entry)
        for entry in entries:
            cat = entry.get("category")
            if cat == "Picture":
                continue
            if cat == "Table":
                render_table(
                    ctx,
                    entry,
                    pictures_for_table=pics_per_table.get(id(entry)),
                    orig_bbox=pre_reflow_table_bboxes.get(id(entry)),
                )
            elif cat == "Formula":
                render_formula(ctx, entry)
            else:
                render_text_entry(ctx, entry)
        # Floating fallback for contained pictures with no image_obj.
        for entry in contained_no_image:
            render_standalone_picture(ctx, entry)

        ctx.flush()
        shape_counter = ctx.next_id + 1

    doc.save(output_path)
    if isinstance(output_path, BytesIO):
        return output_path
    print(f"DOCX file saved to: {output_path}")
    return output_path