"""Table rendering: HTML / Markdown table parsing into a cell grid that
honours colspan/rowspan, then OOXML emission with:
- column widths inferred from cell content + any embedded pictures
- row heights normalised so the table matches its source bbox dimensions
- per-cell font fit to avoid clipping in narrow columns
- inline picture placement inside the matching cell
- fallback to a plain text box when the HTML parse yields no rows
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from .geometry import EMU_PER_PT, bbox_px_to_emu
from .ooxml import (
    NS_W,
    build_anchored_textbox_xml, build_paragraph_xml, build_run_xml,
    build_inline_picture_xml, add_image_relationship,
)
from .text_fit import (
    fit_multiline, get_font, is_cjk_char, wrap_to_width,
)


class TableHTMLParser(HTMLParser):
    """Parses <table>/<tr>/<th>/<td> + colspan/rowspan.

    Each cell is captured as (text, colspan, rowspan) — defaults 1/1 when the
    attribute is missing or malformed. Caller can then build a proper OOXML
    gridSpan / vMerge layout to honour merged header rows seen in the source
    PDFs (very common in CJK technical reports).
    """
    def __init__(self):
        super().__init__()
        self.tables = []
        self.current_table = []
        self.current_row: List[Tuple[str, int, int]] = []
        self.current_cell = ""
        self.current_colspan = 1
        self.current_rowspan = 1
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.in_header = False

    @staticmethod
    def _attr_int(attrs, name: str, default: int = 1) -> int:
        for k, v in attrs:
            if k == name and v:
                try:
                    n = int(v)
                    return n if n > 0 else default
                except (TypeError, ValueError):
                    return default
        return default

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.current_table = []
        elif tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in ("td", "th"):
            self.in_cell = True
            self.current_cell = ""
            self.current_colspan = self._attr_int(attrs, "colspan", 1)
            self.current_rowspan = self._attr_int(attrs, "rowspan", 1)
            if tag == "th":
                self.in_header = True
        elif tag == "br" and self.in_cell:
            self.current_cell += "\n"

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
            if self.current_table:
                self.tables.append(self.current_table)
        elif tag == "tr":
            self.in_row = False
            if self.current_row:
                self.current_table.append((self.current_row, self.in_header))
            self.in_header = False
        elif tag in ("td", "th"):
            self.in_cell = False
            self.current_row.append((
                self.current_cell.strip(),
                self.current_colspan,
                self.current_rowspan,
            ))
            self.current_cell = ""
            self.current_colspan = 1
            self.current_rowspan = 1

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data

    def finalize(self):
        """Flush any cell/row/table left open because the HTML was truncated.

        OCR/VLM output is frequently cut off mid-table (no closing ``</td>`` /
        ``</tr>`` / ``</table>``). Without this, an unterminated ``<table>`` is
        dropped entirely and the whole table silently falls back to raw-text
        rendering. We close the open cell, row and table in order so the
        content is still recovered as a table.
        """
        if self.in_cell:
            self.current_row.append((
                self.current_cell.strip(),
                self.current_colspan,
                self.current_rowspan,
            ))
            self.in_cell = False
            self.current_cell = ""
        if self.current_row:
            self.current_table.append((self.current_row, self.in_header))
            self.current_row = []
        if self.current_table and self.current_table not in self.tables:
            self.tables.append(self.current_table)
            self.current_table = []


# Sanity bounds. OCR/VLM output on unparseable pages (rotated CAD drawings,
# scans) sometimes hallucinates tables with hundreds of near-identical rows or
# dozens of repeated header columns. These caps keep such garbage bounded so it
# can't overflow the page or explode render time; they're far above any real
# table seen in these documents.
MAX_TABLE_ROWS = 200
MAX_TABLE_COLS = 40


def parse_html_table_rows(
    html_string: str,
) -> List[Tuple[List[Tuple[str, int, int]], bool]]:
    """Return a flat list of (cells, is_header) pairs across all tables found.

    Each cell is `(text, colspan, rowspan)`. Robust to truncated HTML (missing
    closing tags) and to runaway/degenerate OCR output: consecutive byte-
    identical rows are collapsed and the total row count is capped.
    """
    parser = TableHTMLParser()
    try:
        parser.feed(html_string)
    except Exception:
        pass
    parser.finalize()   # recover any table left open by truncated HTML

    rows: List[Tuple[List[Tuple[str, int, int]], bool]] = []
    prev_sig = None
    for table in parser.tables:
        for row, is_header in table:
            # Collapse runs of identical rows (a common hallucination shape).
            sig = tuple((t, cs, rs) for (t, cs, rs) in row)
            if sig == prev_sig:
                continue
            prev_sig = sig
            rows.append((row, is_header))
            if len(rows) >= MAX_TABLE_ROWS:
                return rows
    return rows


def parse_markdown_table(md_text: str) -> Optional[List[List[str]]]:
    lines = md_text.strip().split("\n")
    rows = []
    for line in lines:
        if re.match(r"^[\|\s:\-]+$", line):
            continue
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells and any(c for c in cells):
            rows.append(cells)
    return rows or None


def _display_width(s: str) -> int:
    n = 0
    for ch in (s or ""):
        if is_cjk_char(ch):
            n += 2
        else:
            n += 1
    return n


def _longest_token_width(s: str) -> int:
    """Display width of the longest unbreakable run in `s`.

    CJK breaks freely between chars (so a long CJK string isn't an
    unbreakable token), but Latin words and digit runs can't be broken
    mid-word — those set a min column width below which the word would
    visibly overflow or force odd wrapping. Newlines from <br> also force
    a break, so split on them.
    """
    if not s:
        return 0
    best = 0
    for line in s.split("\n"):
        run = 0
        for ch in line:
            if is_cjk_char(ch) or ch.isspace():
                if run > best:
                    best = run
                run = 0
            else:
                run += 1
        if run > best:
            best = run
    return best


def _normalize_column_granularity(
    rows: List[Tuple[List[Tuple[str, int, int]], bool]],
) -> List[Tuple[List[Tuple[str, int, int]], bool]]:
    """Re-span under-segmented rows onto the table body's column boundaries.

    OCR/VLM output for merged-cell tables is often inconsistent: the body rows
    encode each logical column as several grid columns via ``colspan`` (e.g.
    every cell ``colspan=2``), but the header row — or a stray name cell — is
    emitted as single-width ``<td>``s, sometimes with empty separator ``<td>``s
    between them. Placed as-is, those finer cells land on the wrong grid columns
    and the header stops lining up with the body (and, because the leading cells
    are too narrow, downstream rowspan cells get shifted a column over too).

    We fix this BEFORE grid placement so the correction propagates: rows that
    are narrower than the table's full width get their non-empty cells re-spanned
    across the body's canonical column boundaries, absorbing the empty artifact
    cells.

    Crucially the re-span is ROWSPAN-AWARE. A narrow row is not always a full
    row minus artifacts — it can be a sub-header sitting *under* a ``colspan``
    header while the outer columns are already covered by ``rowspan`` cells from
    the row above (e.g. a ``样品组别 / 旋入扭矩 / [失效扭矩 spanning 扭矩值|失效模式] /
    比值`` header block). There the narrow row's cells belong only in the free
    middle columns, NOT stretched across the whole width. We therefore place
    rows top-to-bottom tracking rowspan occupancy, and re-span each narrow row's
    content cells across only the canonical segments still FREE in that row. If
    the content-cell count doesn't match the free-segment count, we leave the
    row untouched rather than guess.

    This is a strict no-op for well-formed tables — uniform-width tables,
    all-``colspan`` tables with nothing under-segmented, ragged data tables
    whose short rows have genuine trailing empties, and sub-header rows that
    already sit correctly under a rowspan block are all left byte-identical.

    Returns the SAME ``rows`` object when nothing changed, so callers can detect
    the no-op cheaply.
    """
    def _row_width(cells):
        return sum(max(1, cs) for (_t, cs, _rs) in cells)

    def _boundaries(cells):
        b = {0}
        col = 0
        for (_t, cs, _rs) in cells:
            col += max(1, cs)
            b.add(col)
        return b

    widths = [_row_width(cells) for cells, _ in rows]
    if not widths:
        return rows
    maxw = max(widths)

    # The body granularity is defined by the rows that fill the full width. Need
    # a stable majority (>= 2) so one over-segmented outlier can't define it.
    full = [cells for (cells, _), w in zip(rows, widths) if w == maxw]
    if len(full) < 2:
        return rows

    # Canonical boundaries: column edges present in EVERY full-width row.
    canon = set.intersection(*[_boundaries(c) for c in full])
    # If the body is already one-cell-per-column there's no coarser grid to
    # align a finer row to — nothing to normalize.
    if canon == set(range(maxw + 1)):
        return rows
    bounds = sorted(canon)          # e.g. [0, 2, 4, 6]
    n_seg = len(bounds) - 1

    # Track rowspan occupancy exactly as parse_table_grid will: place each row's
    # (possibly rewritten) cells into a sparse grid so later rows can see which
    # columns are already taken by a rowspan descending from above.
    occ = [[False] * maxw for _ in range(len(rows))]

    new_rows: List[Tuple[List[Tuple[str, int, int]], bool]] = []
    changed = False
    for row_idx, ((cells, is_header), w) in enumerate(zip(rows, widths)):
        emit_cells = cells

        if w < maxw:
            # Which canonical segments are FREE in this row (no column covered
            # by a rowspan from above)? A segment must be wholly free to count;
            # a segment straddling free + occupied columns is ambiguous -> bail.
            free_segs: List[int] = []
            straddle = False
            for si in range(n_seg):
                cols = range(bounds[si], bounds[si + 1])
                covered = [occ[row_idx][c] for c in cols]
                if not any(covered):
                    free_segs.append(si)
                elif not all(covered):
                    straddle = True
                    break

            if not straddle:
                content = [
                    (t, rs) for (t, _cs, rs) in cells if (t or "").strip()
                ]
                # Re-span content cells across the free segments, one per
                # segment. Only when the counts line up exactly — otherwise we
                # can't unambiguously map cells to segments, so leave as-is.
                if content and len(content) == len(free_segs):
                    rebuilt = []
                    for (t, rs), si in zip(content, free_segs):
                        b0, b1 = bounds[si], bounds[si + 1]
                        rebuilt.append((t, max(1, b1 - b0), rs))
                    if rebuilt != list(cells):
                        emit_cells = rebuilt
                        changed = True

        new_rows.append((emit_cells, is_header))

        # Place emit_cells into `occ` (skipping already-occupied columns) so
        # the rowspans this row starts are visible to subsequent rows.
        col = 0
        for (_t, cs, rs) in emit_cells:
            while col < maxw and occ[row_idx][col]:
                col += 1
            if col >= maxw:
                break
            cs = min(max(1, cs), maxw - col)
            rs = min(max(1, rs), len(rows) - row_idx)
            for rr in range(row_idx, row_idx + rs):
                for cc in range(col, col + cs):
                    occ[rr][cc] = True
            col += cs

    return new_rows if changed else rows


def parse_table_grid(
    rows: List[Tuple[List[Tuple[str, int, int]], bool]],
) -> Tuple[
    int,
    int,
    Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    List[List[bool]],
]:
    """Walk parsed (cells, is_header) rows into a sparse grid.

    Returns (max_cols, n_rows, cell_anchors, occupied).

    The grid width is the MODAL row width, not the maximum. OCR frequently
    mis-segments a single row (an extra spurious split, a stray colspan) so
    that one row is wider than all the others; taking the max would let that
    outlier stretch every other row and shift merged headers out of alignment.
    Using the most common width instead is robust to such outliers. As a
    safeguard we never make the grid narrower than the widest row's last
    NON-EMPTY cell, so real content is never clipped — only trailing empty
    padding from an over-segmented row is dropped. A hard cap keeps
    hallucinated many-column headers bounded.
    """
    from collections import Counter

    # Repair header/body column-granularity mismatch from OCR under-segmentation
    # before placing cells, so the correction propagates to rowspan placement.
    rows = _normalize_column_granularity(rows)

    n_rows = len(rows)

    def _row_width(row_cells):
        return sum(max(1, cs) for (_t, cs, _rs) in row_cells)

    widths = [_row_width(r[0]) for r in rows]
    raw_max = min(max(widths, default=1) or 1, MAX_TABLE_COLS)

    def _place(target_cols: int):
        """Place cells into a target-width grid; report last non-empty col."""
        anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]] = {}
        occ = [[False] * target_cols for _ in range(n_rows)]
        last_nonempty = 0
        for row_idx, (cells, is_header) in enumerate(rows):
            col = 0
            for (cell_text, cs, rs) in cells:
                while col < target_cols and occ[row_idx][col]:
                    col += 1
                if col >= target_cols:
                    break
                cs = min(max(1, cs), target_cols - col)
                rs = min(max(1, rs), n_rows - row_idx)
                anchors[(row_idx, col)] = (cell_text, cs, rs, is_header)
                for rr in range(row_idx, row_idx + rs):
                    for cc in range(col, col + cs):
                        occ[rr][cc] = True
                if (cell_text or "").strip():
                    last_nonempty = max(last_nonempty, col + cs)
                col += cs
        return anchors, occ, last_nonempty

    # First pass at the widest observed width to learn where real content ends.
    _a0, _o0, content_extent = _place(raw_max)
    # Modal width, but never below the real-content extent, never above the cap.
    modal = Counter(widths).most_common(1)[0][0] if widths else 1
    max_cols = max(1, min(raw_max, max(modal, content_extent)))

    cell_anchors, occupied, _ = _place(max_cols)
    return max_cols, n_rows, cell_anchors, occupied


def _assign_pictures_to_cells(
    pictures: List[Dict],
    bbox: List[int],
    col_weight: List[int],
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    max_cols: int,
    n_rows: int,
) -> Dict[Tuple[int, int], List[Dict]]:
    """Decide which (row, col) anchor each picture lives in.

    Why not just bbox-centroid → column edges? Two failure modes that hit
    real PDFs:
      1. One column has very long text and steals geometric width from a
         neighbouring image column (`备注` cell expands past the picture).
      2. The dotsocr layout returns pictures in cells the OCR labelled
         empty (the cell text is `""`), so emptiness is a strong signal
         that's lost if we only look at x-positions.

    The scoring combines:
      * row band: rows are split into equal vertical bands inside the
        table bbox (we don't know real row heights yet — column-weight
        widths are derived first). Each picture lands in the row whose
        band contains its y-centroid.
      * cell emptiness in that row: empty cells score highest, short
        labels (≤ 3 chars) next, long text cells last.
      * horizontal proximity: among candidate cells with the best
        emptiness tier, the cell whose column midpoint is closest to the
        picture x-centroid wins.

    The returned mapping keys on the (row, col) of the CELL ANCHOR
    (rowspan/colspan accounted for), so downstream code can index into
    `cell_anchors` directly.
    """
    out: Dict[Tuple[int, int], List[Dict]] = {}
    if not pictures or n_rows <= 0 or max_cols <= 0:
        return out

    tbl_x1, tbl_y1, tbl_x2, tbl_y2 = bbox
    tbl_w_px = max(1.0, tbl_x2 - tbl_x1)
    tbl_h_px = max(1.0, tbl_y2 - tbl_y1)
    # Pre-row vertical band: we don't have real row heights yet, so split
    # the bbox into equal-height bands. This is good enough to identify the
    # row a picture lives in for the dominant "image-in-its-own-row" case.
    row_band_h = tbl_h_px / n_rows
    # Text-derived column boundaries (same formula as the live rendering
    # below — we only use them as a tiebreaker for horizontal proximity).
    weight_total = sum(col_weight) or max_cols
    col_edges_px = [tbl_x1]
    for cw in col_weight:
        col_edges_px.append(col_edges_px[-1] + tbl_w_px * (cw / weight_total))

    # For each row, list every cell anchor whose VERTICAL SPAN covers that
    # row — including anchors that started earlier and continue via rowspan.
    # This lets a picture sitting in a rowspan-merged cell pick the anchor
    # at the rowspan's top, not a numeric continuation cell that happens to
    # land in the same equal-height row band.
    cells_by_row: Dict[int, List[Tuple[int, int, int, str]]] = {}
    for (r, c), (txt, cs, rs, _h) in cell_anchors.items():
        for rr in range(r, min(n_rows, r + max(1, rs))):
            cells_by_row.setdefault(rr, []).append((r, c, cs, txt or ""))

    def _emptiness_tier(text: str) -> int:
        """0 = empty, 1 = very short label, 2 = anything longer.
        Lower tier wins (more likely to hold a picture)."""
        s = (text or "").strip()
        if not s:
            return 0
        # CJK display width counts double; keep the threshold conservative
        if _display_width(s) <= 6:
            return 1
        return 2

    for pic in pictures:
        pb = pic.get("bbox") or []
        if len(pb) != 4:
            continue
        cx_px = (pb[0] + pb[2]) / 2.0
        cy_px = (pb[1] + pb[3]) / 2.0

        # Row band: clamp into [0, n_rows-1]
        row_idx = int((cy_px - tbl_y1) / row_band_h)
        row_idx = max(0, min(n_rows - 1, row_idx))

        # Walk rows outward from row_idx looking for the row whose anchors
        # have a candidate cell. The picture may visually overlap two row
        # bands, so we don't insist on the band match alone.
        candidates: List[Tuple[int, int, int, float]] = []
        for dr in range(n_rows):  # 0, then 1, -1, 2, -2, ...
            offsets = (0,) if dr == 0 else (dr, -dr)
            for off in offsets:
                r = row_idx + off
                if r < 0 or r >= n_rows:
                    continue
                row_cells = cells_by_row.get(r) or []
                for (anchor_r, c, cs, txt) in row_cells:
                    tier = _emptiness_tier(txt)
                    # Span midpoint accounts for colspan>1 cells.
                    mid = (col_edges_px[c] + col_edges_px[min(max_cols, c + cs)]) / 2.0
                    dist = abs(cx_px - mid)
                    candidates.append((anchor_r, c, tier, dist))
                if candidates:
                    break
            if candidates:
                break

        if not candidates:
            continue
        # Best = lowest tier, then lowest x-distance.
        candidates.sort(key=lambda t: (t[2], t[3]))
        anchor_r, anchor_c, _tier, _dist = candidates[0]
        out.setdefault((anchor_r, anchor_c), []).append(pic)
    return out


def compute_col_weights(
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    max_cols: int,
) -> List[int]:
    """Per-column weights inferred from cell text.

    A column's soft weight is the MEDIAN of its cells' width contributions,
    not the maximum. Using the max let a single outlier cell — e.g. one matrix
    cell holding a long ``H1, H2, … H15`` list among otherwise empty cells, or
    one long paragraph in an otherwise short column — blow that column out to
    many times its neighbours' width, squeezing the rest of the table. The
    median keeps a column wide only when it is *consistently* wide (a real
    description column, long in every row) and lets a lone long cell wrap
    instead. A hard floor from the longest unbreakable token is still applied
    per column (via max) so no column is squeezed below a word/code it must
    contain and would otherwise clip.
    """
    from statistics import median

    contribs: List[List[int]] = [[] for _ in range(max_cols)]
    col_min = [1] * max_cols
    for (_r, c), (txt, cs, _rs, _h) in cell_anchors.items():
        if c >= max_cols:
            continue
        cs = max(1, min(cs, max_cols - c))
        if cs == 1:
            contribs[c].append(_display_width(txt))
            col_min[c] = max(col_min[c], _longest_token_width(txt))
        else:
            share = max(1, _display_width(txt) // cs)
            tok_share = max(1, _longest_token_width(txt) // cs)
            for cc in range(c, c + cs):
                contribs[cc].append(share)
                col_min[cc] = max(col_min[cc], tok_share)

    col_weight = []
    for c in range(max_cols):
        # Median of the non-trivial contributions, so empty placeholder cells
        # don't drag a genuinely-wide column down, and a lone long cell doesn't
        # inflate an otherwise-empty column.
        nontrivial = [v for v in contribs[c] if v > 1]
        col_weight.append(int(median(nontrivial)) if nontrivial else 1)

    # Lift each column to its longest-unbreakable-token floor so narrow-but-
    # text-bearing columns (e.g. an 'N/ACC' column) stay readable.
    return [max(w, m) for w, m in zip(col_weight, col_min)]


def render_table(
    ctx,
    entry: Dict,
    pictures_for_table: Optional[List[Dict]] = None,
) -> None:
    text = (entry.get("text") or "").strip()
    if not text:
        return
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    x, y, w, h = bbox_px_to_emu(
        bbox, ctx.zoom, ctx.page_w_pt, ctx.page_h_pt,
    )

    # Parse table cells (HTML preferred; markdown fallback). Cells carry
    # (text, colspan, rowspan) so we can honour merged headers.
    rows: List[Tuple[List[Tuple[str, int, int]], bool]] = []
    if "<table" in text.lower():
        rows = parse_html_table_rows(text)
    else:
        md_rows = parse_markdown_table(text)
        if md_rows:
            rows = [([(c, 1, 1) for c in r], False) for r in md_rows]

    style = entry.get("style") or {}

    if not rows:
        # Fall back to text rendering inside a positioned box.
        x1, y1, x2, y2 = bbox
        box_w_pt = max(1.0, (x2 - x1) / ctx.zoom)
        box_h_pt = max(1.0, (y2 - y1) / ctx.zoom)
        t_style = dict(style)
        base_size_pt = float(t_style.get("size") or 11.0)
        fitted, lines = fit_multiline(
            text, box_w_pt, box_h_pt, max_size_pt=base_size_pt,
        )
        if fitted < base_size_pt:
            t_style["size"] = fitted
        line_pt = None
        render_text = text
        if lines:
            render_text = "\n".join(lines)
            if len(lines) >= 2:
                line_pt = box_h_pt / len(lines)
        runs_xml = build_run_xml(render_text, t_style)
        para_xml = build_paragraph_xml(runs_xml, line_pt=line_pt)
        ctx.xml_chunks.append(
            build_anchored_textbox_xml(
                x, y, w, h, para_xml, ctx._next_id(),
                body_auto_fit=True,
            )
        )
        return

    max_cols, n_rows, cell_anchors, occupied = parse_table_grid(rows)

    # Column widths: inferred from content, normalized to bbox width so the
    # total table width = source bbox width. When the entry carries
    # `_shared_col_weights` (set by the chain detector in json_to_docx for
    # tables that continue across pages), prefer those — that keeps the
    # column proportions identical across the head and continuation rows of
    # the same logical table.
    shared = entry.get("_shared_col_weights")
    if shared and len(shared) == max_cols:
        col_weight = [max(1, int(w)) for w in shared]
    else:
        col_weight = compute_col_weights(cell_anchors, max_cols)

    # Weight columns by any pictures contained in the table. The cell-based
    # detector (`_assign_pictures_to_cells`, used later for inline placement)
    # is the source of truth for "which cell does this picture belong to" —
    # it scores each candidate cell by emptiness + horizontal proximity, so
    # it works even when a text-heavy neighbouring column would otherwise
    # swallow the picture under naïve centroid-in-edges geometry.
    #
    # We pre-compute that assignment here so the column it picks can be
    # weight-boosted; then the same map is reused inline below to actually
    # place the picture XML.
    pic_assignments: Dict[Tuple[int, int], List[Dict]] = {}
    # Per-column minimum width (pt) demanded by any picture assigned to that
    # column. Zero for text-only columns / picture-free tables.
    pic_col_min_pt = [0.0] * max_cols
    if pictures_for_table:
        pic_assignments = _assign_pictures_to_cells(
            pictures_for_table, bbox, col_weight, cell_anchors,
            max_cols, n_rows,
        )
        # A picture column's need is a MINIMUM WIDTH (the fitted image width),
        # not a proportional weight. Record the widest picture width per column
        # here and fold it into the per-column floor below. Expressing it as a
        # weight instead (the old behaviour) let the image column's huge pixel
        # count dominate the proportional split and squeeze every numeric
        # column down to a few points.
        zoom = max(1e-6, float(getattr(ctx, "zoom", 1.0) or 1.0))
        for (_r, c), pics in pic_assignments.items():
            for pic in pics:
                pb = pic.get("bbox") or []
                if len(pb) != 4 or c >= max_cols:
                    continue
                pic_w_pt = max(1.0, (pb[2] - pb[0]) / zoom)
                pic_col_min_pt[c] = max(pic_col_min_pt[c], pic_w_pt)

    # Allocate width with a floor per column: a column with a short label
    # like 'P3' still needs enough room for that label, otherwise narrow
    # columns get squeezed to a few millimetres and the text either clips
    # or wraps awkwardly. The floor is the larger of: ~4 chars at the
    # base font size, or the column's longest unbreakable token. Floors are
    # capped so they can never consume more than 60% of the table — that
    # would leave nothing for the long-text columns.
    declared_size_pt = float(style.get("size") or 11.0)
    char_w_pt = max(4.0, declared_size_pt * 0.55)
    min_col_pt = char_w_pt * 4.0  # ~"N/ACC" width with padding
    min_col_emu_default = int(round(min_col_pt * EMU_PER_PT))
    # Cap the floor so the floors collectively never exceed 60% of `w` —
    # otherwise a wide-column-heavy table would have nothing left to spend.
    min_col_cap = max(1, int(w * 0.6 / max_cols))
    floor_per_col = min(min_col_emu_default, min_col_cap)
    min_col_emus = [floor_per_col] * max_cols
    # Raise the floor of any picture column to the picture's own width (plus a
    # little padding), so the column is guaranteed wide enough to show the
    # image at full size without upscaling the whole column out of proportion.
    # Cap a single picture column at 55% of the table so the data columns keep
    # a usable share.
    if any(pic_col_min_pt):
        pic_col_cap_emu = int(round(w * 0.55))
        for c in range(max_cols):
            if pic_col_min_pt[c] > 0:
                need_emu = int(round((pic_col_min_pt[c] + 4.0) * EMU_PER_PT))
                min_col_emus[c] = max(
                    min_col_emus[c], min(need_emu, pic_col_cap_emu)
                )

    total_weight = sum(col_weight) or max_cols
    raw_emus = [
        max(1, int(round(w * cw / total_weight))) for cw in col_weight
    ]
    # Lift columns to their floor, then re-balance: take the over-allocation
    # proportionally from columns that are above their floor.
    col_w_emus = [max(rw, mn) for rw, mn in zip(raw_emus, min_col_emus)]
    delta = sum(col_w_emus) - w
    if delta > 0:
        # Trim from columns that have headroom above their floor.
        for _ in range(8):  # at most a few passes — converges fast
            slack = [c - mn for c, mn in zip(col_w_emus, min_col_emus)]
            slack_total = sum(s for s in slack if s > 0)
            if slack_total <= 0 or delta <= 0:
                break
            new_emus = []
            for c, s in zip(col_w_emus, slack):
                if s <= 0:
                    new_emus.append(c)
                    continue
                take = int(round(delta * s / slack_total))
                take = min(take, s)
                new_emus.append(c - take)
            new_delta = sum(new_emus) - w
            if new_delta == delta:
                break
            col_w_emus = new_emus
            delta = new_delta
    # Final fixup so the row sums exactly to `w`.
    drift = sum(col_w_emus) - w
    if drift != 0:
        # Adjust the widest column to absorb the rounding drift.
        widest = max(range(max_cols), key=lambda i: col_w_emus[i])
        col_w_emus[widest] = max(1, col_w_emus[widest] - drift)

    grid_xml = "<w:tblGrid>" + ("".join(
        f'<w:gridCol w:w="{cw}"/>' for cw in col_w_emus
    )) + "</w:tblGrid>"

    borders_xml = (
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '</w:tblBorders>'
    )
    tbl_pr_xml = (
        '<w:tblPr>'
        f'<w:tblW w:w="{w}" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        f'{borders_xml}'
        '</w:tblPr>'
    )

    # Per-cell font cap based on the narrowest column to keep cells readable.
    narrowest_col_pt = min(col_w_emus) / EMU_PER_PT
    size_ceiling_pt = min(
        declared_size_pt, max(7.0, narrowest_col_pt / 5.0), 12.0,
    )
    cell_size_pt = size_ceiling_pt
    cell_size_px = max(1, int(round(cell_size_pt)))
    meas_font = get_font(cell_size_px)
    natural_h_px = (
        (sum(meas_font.getmetrics()) if meas_font else cell_size_px * 1.2)
    )

    def _line_count_in_col(cell: str, col_idx: int, cs: int = 1) -> int:
        if not cell or meas_font is None:
            return 1
        cell_w_emu = sum(col_w_emus[col_idx:col_idx + max(1, cs)])
        cell_w_pt = cell_w_emu / EMU_PER_PT
        wrapped = wrap_to_width(
            cell, meas_font, max(1.0, cell_w_pt * 0.95), cell_size_px,
        )
        return max(1, len(wrapped))

    row_lines_arr = [1] * n_rows
    for (row_idx, col_idx), (cell_text, cs, _rs, _is_h) in cell_anchors.items():
        row_lines_arr[row_idx] = max(
            row_lines_arr[row_idx],
            _line_count_in_col(cell_text, col_idx, cs),
        )

    bbox_h_pt = h / EMU_PER_PT
    weight_sum = sum(row_lines_arr) or n_rows or 1
    row_h_pts = [
        (bbox_h_pt * rl / weight_sum) for rl in row_lines_arr
    ]
    # Floor each row by ITS OWN line count, not a flat one-line minimum —
    # otherwise a row that wraps to 2+ lines (long text, narrow column) can be
    # squeezed by the proportional bbox split down to one line's height, and
    # since rows are emitted with hRule="exact" (hard clip, no autofit), the
    # extra line(s) render past the row border and get visually clipped by
    # the next row's border line.
    line_h_pt = natural_h_px * 1.06
    row_h_pts = [
        max(rh, line_h_pt * rl) for rh, rl in zip(row_h_pts, row_lines_arr)
    ]

    # Reserve vertical room for pictures. A picture cell (often a rowspan block
    # holding one or more stacked illustrations) needs its merged height to be
    # at least the SUM of its pictures' fitted heights, else the images spill
    # past the cell border (rows use hRule="exact", a hard clip). We compute
    # each picture's height when scaled to the cell's column width, add a small
    # per-picture pad, and if the anchor's rowspan rows don't already provide
    # that much height, grow those rows evenly to cover the deficit. Text-only
    # cells and picture-free tables are untouched.
    #
    # Per-row minimum kept for the bbox re-fit below: a row must never shrink
    # below its own text line count, otherwise text clips.
    row_min_pts = [line_h_pt * rl for rl in row_lines_arr]
    # Height (EMU) each picture cell reserves for its images, so the per-cell
    # text fit below can subtract it and fit the label into the space above.
    _pic_reserve_emu_by_cell: Dict[Tuple[int, int], int] = {}
    if pic_assignments:
        pic_pad_pt = 3.0
        for (a_r, a_c), pics in pic_assignments.items():
            anchor = cell_anchors.get((a_r, a_c))
            if not anchor:
                continue
            _t, a_cs, a_rs, _h = anchor
            col_w_pt = sum(
                col_w_emus[a_c:a_c + max(1, a_cs)]
            ) / EMU_PER_PT
            avail_w_pt = max(1.0, col_w_pt - 4.0)   # ~2pt padding each side
            need_pt = 0.0
            for pic in pics:
                pb = pic.get("bbox") or []
                if len(pb) != 4 or pb[2] <= pb[0] or pb[3] <= pb[1]:
                    continue
                iw_pt = (pb[2] - pb[0]) / ctx.zoom
                ih_pt = (pb[3] - pb[1]) / ctx.zoom
                # height once the picture is scaled down to fit the column
                # width (never upscaled)
                scale = min(1.0, avail_w_pt / max(1.0, iw_pt))
                need_pt += ih_pt * scale + pic_pad_pt
            _pic_reserve_emu_by_cell[(a_r, a_c)] = int(
                round(need_pt * EMU_PER_PT)
            )
            span = list(range(a_r, min(n_rows, a_r + max(1, a_rs))))
            have_pt = sum(row_h_pts[r] for r in span)
            deficit = need_pt - have_pt
            if deficit > 0:
                add = deficit / len(span)
                for r in span:
                    row_h_pts[r] += add
                    # A picture row's minimum is now the taller of its text
                    # floor and its share of the picture demand, so the bbox
                    # re-fit below can't squeeze the image back out.
                    row_min_pts[r] = max(row_min_pts[r], row_h_pts[r])

    # Keep the table inside its source bbox height. Growing rows for pictures
    # (or wrapped text) can push the total past the bbox; because the table
    # lives in a fixed-height anchored textbox (noAutofit), any overflow is
    # hard-clipped — which drops the bottom rows and can also overlap whatever
    # sits just below the table on the page. So if we're over budget, reclaim
    # the excess from rows that have slack ABOVE their minimum, proportionally,
    # leaving every row at least its minimum (text + any picture share). If the
    # minimums themselves already exceed the bbox (a genuinely oversized table),
    # scale everything down uniformly so all rows stay visible and none clip.
    total_pt = sum(row_h_pts)
    if total_pt > bbox_h_pt + 0.5:
        min_total = sum(row_min_pts)
        if min_total <= bbox_h_pt:
            excess = total_pt - bbox_h_pt
            for _ in range(8):
                slack = [rh - mn for rh, mn in zip(row_h_pts, row_min_pts)]
                slack_total = sum(s for s in slack if s > 0)
                if slack_total <= 1e-6 or excess <= 0.5:
                    break
                new_h = []
                for rh, s in zip(row_h_pts, slack):
                    if s <= 0:
                        new_h.append(rh)
                        continue
                    take = min(s, excess * s / slack_total)
                    new_h.append(rh - take)
                excess = sum(new_h) - bbox_h_pt
                row_h_pts = new_h
        else:
            # Even the minimums don't fit — scale uniformly so nothing clips.
            scale = bbox_h_pt / total_pt
            row_h_pts = [rh * scale for rh in row_h_pts]

    # Picture placement reuses the cell assignment already computed above
    # (the same one that fed into col_weight). Recomputing here from final
    # `col_w_emus` would diverge from the placement we promised the weight
    # boost was based on, and the assignment doesn't need final widths
    # because it scores by emptiness + horizontal proximity, not strict
    # column boundaries.
    pic_inline_by_cell: Dict[Tuple[int, int], List[Dict]] = pic_assignments

    # Cell + row emission.
    rows_xml_parts: List[str] = []
    for row_idx in range(n_rows):
        cells_xml: List[str] = []
        skip_cols = 0   # columns consumed by a preceding cell's gridSpan
        for col_idx in range(max_cols):
            if skip_cols > 0:
                skip_cols -= 1
                continue
            anchor = cell_anchors.get((row_idx, col_idx))
            if anchor is not None:
                cell_text, cs, rs, is_header = anchor
                cell_w_emu = sum(col_w_emus[col_idx:col_idx + cs])
                pics_here = pic_inline_by_cell.get((row_idx, col_idx)) or []
                pic_paragraphs = []
                cell_h_emu = int(round(row_h_pts[row_idx] * EMU_PER_PT))
                if rs > 1:
                    cell_h_emu = sum(
                        int(round(row_h_pts[row_idx + dr] * EMU_PER_PT))
                        for dr in range(rs)
                    )

                # Per-cell fit so text is FULLY VISIBLE. Each cell is re-fitted
                # against its final (cell_w x cell_h). The base size is the
                # table-wide cap `cell_size_pt` (from the narrowest column);
                # `fit_multiline` shrinks further inside that budget and
                # truncates with `…` when even the floor doesn't fit, so cell
                # content never overflows the fixed row height (`hRule="exact"`).
                # When the cell also holds pictures, fit the text into the space
                # left ABOVE them so text and image don't collide.
                cell_w_pt_for_fit = cell_w_emu / EMU_PER_PT
                cell_h_pt_for_fit = cell_h_emu / EMU_PER_PT
                if any(p.get("image_obj") for p in pics_here):
                    reserve_pt = (
                        _pic_reserve_emu_by_cell.get((row_idx, col_idx), 0)
                        / EMU_PER_PT
                    )
                    one_line_pt = natural_h_px * 1.06
                    cell_h_pt_for_fit = max(
                        one_line_pt, cell_h_pt_for_fit - reserve_pt,
                    )
                fitted_cell_pt, fitted_cell_lines = fit_multiline(
                    cell_text,
                    max(1.0, cell_w_pt_for_fit),
                    max(1.0, cell_h_pt_for_fit),
                    max_size_pt=cell_size_pt,
                )
                cell_style = dict(style)
                cell_style["size"] = (
                    fitted_cell_pt if cell_text else cell_size_pt
                )
                if is_header or row_idx == 0:
                    cell_style["bold"] = True
                rendered_cell_text = (
                    "\n".join(fitted_cell_lines)
                    if fitted_cell_lines else cell_text
                )
                run_xml = build_run_xml(rendered_cell_text, cell_style)
                # Tables are universally centered in the source docs we see
                # (engineering reports, risk matrices, parts lists). Center
                # horizontally + vertically so cells read consistently and a
                # short value in a wide column isn't pinned to the left edge.
                para_xml = build_paragraph_xml(run_xml, alignment="center")
                text_reserve_emu = (
                    int(round(natural_h_px * 1.06 * EMU_PER_PT))
                    if cell_text else 0
                )
                pic_budget_h_emu = max(1, cell_h_emu - text_reserve_emu)
                pad_emu = 2 * EMU_PER_PT
                max_pic_w_emu = max(1, cell_w_emu - pad_emu)
                n_pics = sum(1 for p in pics_here if p.get("image_obj"))
                per_pic_h_emu = (
                    max(1, pic_budget_h_emu // n_pics) if n_pics else 1
                )
                for pic in pics_here:
                    img_obj = pic.get("image_obj")
                    if img_obj is None:
                        continue
                    buf = BytesIO()
                    img_obj.save(buf, format="PNG")
                    buf.seek(0)
                    rel_id = add_image_relationship(ctx.doc, buf, "png")
                    iw, ih = img_obj.size
                    if iw <= 0 or ih <= 0:
                        continue
                    # Intrinsic size in the SAME point space as the cell: the
                    # crop pixels are rendered at ctx.zoom (a 2x page raster is
                    # typical), so divide by zoom to get the picture's true
                    # physical size — exactly what render_standalone_picture
                    # does via bbox_px_to_emu. Prefer the picture's own bbox
                    # when present (already the source-accurate extent); fall
                    # back to the crop pixel dims / zoom. Treating the raw crop
                    # pixels as points made every table image ~zoom-times too
                    # big, so it filled the whole column width and overflowed
                    # the cell height regardless of the fit clamp below.
                    zoom = max(1e-6, float(getattr(ctx, "zoom", 1.0) or 1.0))
                    pb = pic.get("bbox") or []
                    if len(pb) == 4 and pb[2] > pb[0] and pb[3] > pb[1]:
                        intrinsic_w_pt = (pb[2] - pb[0]) / zoom
                        intrinsic_h_pt = (pb[3] - pb[1]) / zoom
                    else:
                        intrinsic_w_pt = iw / zoom
                        intrinsic_h_pt = ih / zoom
                    intrinsic_w_emu = max(1.0, intrinsic_w_pt * EMU_PER_PT)
                    intrinsic_h_emu = max(1.0, intrinsic_h_pt * EMU_PER_PT)
                    # Fit inside the cell: never wider than the padded column,
                    # never taller than this picture's share of the cell height,
                    # and never upscale past the source size.
                    scale_w = max_pic_w_emu / intrinsic_w_emu
                    scale_h = per_pic_h_emu / intrinsic_h_emu
                    scale = min(1.0, scale_w, scale_h)
                    pic_w_emu = max(1, int(round(intrinsic_w_emu * scale)))
                    pic_h_emu = max(1, int(round(intrinsic_h_emu * scale)))
                    inline_pic_xml = build_inline_picture_xml(
                        pic_w_emu, pic_h_emu, rel_id, ctx._next_id(),
                        pic_name=pic.get("id") or "Picture",
                    )
                    pic_paragraphs.append(
                        f'<w:p xmlns:w="{NS_W}">'
                        f'<w:pPr><w:spacing w:before="0" w:after="0"/>'
                        f'<w:jc w:val="center"/></w:pPr>'
                        f'{inline_pic_xml}</w:p>'
                    )
                tc_pr_parts = [f'<w:tcW w:w="{cell_w_emu}" w:type="dxa"/>']
                if cs > 1:
                    tc_pr_parts.append(f'<w:gridSpan w:val="{cs}"/>')
                if rs > 1:
                    tc_pr_parts.append('<w:vMerge w:val="restart"/>')
                tc_pr_parts.append('<w:vAlign w:val="center"/>')
                cells_xml.append(
                    "<w:tc>"
                    f'<w:tcPr>{"".join(tc_pr_parts)}</w:tcPr>'
                    f"{para_xml}{''.join(pic_paragraphs)}"
                    "</w:tc>"
                )
            elif occupied[row_idx][col_idx]:
                # Continuation slot: emit vMerge continuation if the cell
                # above is a vMerge anchor; otherwise it's a colspan
                # continuation covered by the previous cell's gridSpan.
                above_is_vmerge = False
                vmerge_cs = 1
                r = row_idx - 1
                while r >= 0:
                    if (r, col_idx) in cell_anchors:
                        _t, _cs, _rs, _h = cell_anchors[(r, col_idx)]
                        if _rs > 1 and r + _rs > row_idx:
                            above_is_vmerge = True
                            vmerge_cs = max(1, _cs)
                        break
                    if not occupied[r][col_idx]:
                        break
                    r -= 1
                if above_is_vmerge:
                    # The vMerge anchor may ALSO span columns (rowspan+colspan,
                    # e.g. a corner header). The continuation cell must repeat
                    # the SAME gridSpan and consume the spanned continuation
                    # columns — otherwise this row covers fewer grid columns
                    # than the others, which makes Word/LibreOffice misalign
                    # the row and drop trailing cells further down the table.
                    cont_w_emu = sum(col_w_emus[col_idx:col_idx + vmerge_cs])
                    span_xml = (
                        f'<w:gridSpan w:val="{vmerge_cs}"/>'
                        if vmerge_cs > 1 else ""
                    )
                    empty_para = build_paragraph_xml("", alignment="center")
                    cells_xml.append(
                        "<w:tc>"
                        f'<w:tcPr><w:tcW w:w="{cont_w_emu}" w:type="dxa"/>'
                        f'{span_xml}'
                        '<w:vMerge w:val="continue"/>'
                        '<w:vAlign w:val="center"/></w:tcPr>'
                        f"{empty_para}"
                        "</w:tc>"
                    )
                    skip_cols = vmerge_cs - 1   # skip the columns we just spanned
                # else: skip — covered by previous cell's gridSpan
            else:
                # No content at this position (uncommon edge case).
                empty_para = build_paragraph_xml("", alignment="center")
                cells_xml.append(
                    "<w:tc>"
                    f'<w:tcPr><w:tcW w:w="{col_w_emus[col_idx]}" w:type="dxa"/>'
                    '<w:vAlign w:val="center"/></w:tcPr>'
                    f"{empty_para}"
                    "</w:tc>"
                )
        # `exact` (not `atLeast`) so a single cell with oversized content
        # can't blow past the row budget and push the table off the source
        # bbox — that's what overlapped the text below the table on page 1.
        # Pictures placed in cells are pre-scaled to fit (see scale_h above),
        # so `exact` is safe.
        row_h_twips = max(1, int(round(row_h_pts[row_idx] * 20)))
        tr_pr = f'<w:trPr><w:trHeight w:val="{row_h_twips}" w:hRule="exact"/></w:trPr>'
        rows_xml_parts.append("<w:tr>" + tr_pr + "".join(cells_xml) + "</w:tr>")

    tbl_xml = f"<w:tbl>{tbl_pr_xml}{grid_xml}{''.join(rows_xml_parts)}</w:tbl>"
    body_xml = tbl_xml + '<w:p><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr></w:p>'
    # noAutofit: don't let the textbox grow past the source bbox. Rows + cell
    # content are all sized to fit so nothing should overflow.
    ctx.xml_chunks.append(
        build_anchored_textbox_xml(
            x, y, w, h, body_xml, ctx._next_id(),
            body_auto_fit=False,
        )
    )