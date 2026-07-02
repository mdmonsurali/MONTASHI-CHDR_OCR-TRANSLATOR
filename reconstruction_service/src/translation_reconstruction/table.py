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


def parse_html_table_rows(
    html_string: str,
) -> List[Tuple[List[Tuple[str, int, int]], bool]]:
    """Return a flat list of (cells, is_header) pairs across all tables found.

    Each cell is `(text, colspan, rowspan)`.
    """
    parser = TableHTMLParser()
    parser.feed(html_string)
    rows: List[Tuple[List[Tuple[str, int, int]], bool]] = []
    for table in parser.tables:
        for row, is_header in table:
            rows.append((row, is_header))
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
    """
    def _row_width(row_cells):
        return sum(max(1, cs) for (_t, cs, _rs) in row_cells)
    max_cols = max((_row_width(r[0]) for r in rows), default=1) or 1
    n_rows = len(rows)
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]] = {}
    occupied: List[List[bool]] = [
        [False] * max_cols for _ in range(n_rows)
    ]
    for row_idx, (cells, is_header) in enumerate(rows):
        col = 0
        for (cell_text, cs, rs) in cells:
            while col < max_cols and occupied[row_idx][col]:
                col += 1
            if col >= max_cols:
                break
            cs = max(1, cs)
            rs = max(1, rs)
            cs = min(cs, max_cols - col)
            rs = min(rs, n_rows - row_idx)
            cell_anchors[(row_idx, col)] = (cell_text, cs, rs, is_header)
            for rr in range(row_idx, row_idx + rs):
                for cc in range(col, col + cs):
                    occupied[rr][cc] = True
            col += cs
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

    Translation breaks the simple "short text = image cell" signal: a 3-CJK
    label like 横截面 expands to "Corte transversal" / "Cross section",
    which then outscores neighbouring numeric cells like `15.5` under any
    length-based tier. Two refinements keep this working:

    1. Per-row scoring picks a candidate (row, col) for each picture using
       emptiness + horizontal proximity, just like before — this still
       resolves the picture-in-its-own-row case (e.g. the `Âncora II`
       rowspan cell that is literally empty).
    2. Pictures whose bboxes share an x-range (same column visually) are
       grouped, and the GROUP's column index is a majority vote across its
       per-picture candidates. So even when one picture's row has only
       numeric neighbours, it inherits the column from siblings whose row
       contained the empty cell.

    The vote is the key fix: a single empty cell anywhere in the picture
    cluster's row range is enough to anchor the whole cluster to that
    column, regardless of what translation did to label text.
    """
    out: Dict[Tuple[int, int], List[Dict]] = {}
    if not pictures or n_rows <= 0 or max_cols <= 0:
        return out

    tbl_x1, tbl_y1, tbl_x2, tbl_y2 = bbox
    tbl_w_px = max(1.0, tbl_x2 - tbl_x1)
    tbl_h_px = max(1.0, tbl_y2 - tbl_y1)
    row_band_h = tbl_h_px / n_rows
    weight_total = sum(col_weight) or max_cols
    col_edges_px = [tbl_x1]
    for cw in col_weight:
        col_edges_px.append(col_edges_px[-1] + tbl_w_px * (cw / weight_total))

    # For each row, list every cell anchor whose VERTICAL SPAN covers that
    # row — anchors at earlier rows that continue via rowspan are included.
    cells_by_row: Dict[int, List[Tuple[int, int, int, str]]] = {}
    for (r, c), (txt, cs, rs, _h) in cell_anchors.items():
        for rr in range(r, min(n_rows, r + max(1, rs))):
            cells_by_row.setdefault(rr, []).append((r, c, cs, txt or ""))

    def _emptiness_tier(text: str) -> int:
        """0 = empty, 1 = very short label, 2 = anything longer."""
        s = (text or "").strip()
        if not s:
            return 0
        if _display_width(s) <= 6:
            return 1
        return 2

    def _score_picture(pic: Dict) -> Tuple[Optional[Tuple[int, int]],
                                            List[Tuple[int, int, int, float]]]:
        """Return (best (row,col), all candidates) for a single picture.

        The 2nd element lets the cluster vote inspect ALL candidates per
        picture, not just the winner — important when one picture's
        empty-cell signal should propagate to siblings whose own row has
        only long label cells after translation.
        """
        pb = pic.get("_orig_bbox") or pic.get("bbox") or []
        if len(pb) != 4:
            return None, []
        cx_px = (pb[0] + pb[2]) / 2.0
        cy_px = (pb[1] + pb[3]) / 2.0
        row_idx = max(0, min(n_rows - 1, int((cy_px - tbl_y1) / row_band_h)))

        candidates: List[Tuple[int, int, int, float]] = []
        for dr in range(n_rows):
            offsets = (0,) if dr == 0 else (dr, -dr)
            for off in offsets:
                r = row_idx + off
                if r < 0 or r >= n_rows:
                    continue
                for (anchor_r, c, cs, txt) in cells_by_row.get(r) or []:
                    tier = _emptiness_tier(txt)
                    mid = (col_edges_px[c] + col_edges_px[min(max_cols, c + cs)]) / 2.0
                    dist = abs(cx_px - mid)
                    candidates.append((anchor_r, c, tier, dist))
                if candidates:
                    break
            if candidates:
                break
        if not candidates:
            return None, []
        candidates.sort(key=lambda t: (t[2], t[3]))
        return (candidates[0][0], candidates[0][1]), candidates

    # Per-picture preferred anchors AND full candidate lists.
    individual: List[Tuple[Dict, Optional[Tuple[int, int]],
                           List[Tuple[int, int, int, float]]]] = []
    for p in pictures:
        best, cands = _score_picture(p)
        individual.append((p, best, cands))

    # Cluster pictures by x-overlap: two pictures are in the same cluster
    # when their bbox x-spans overlap by ≥ 50% of the smaller span. This
    # catches vertically-stacked images of similar width (the common
    # "all images sit in one column" pattern) without flagging unrelated
    # diagrams elsewhere on the page.
    def _x_overlap(a: Dict, b: Dict) -> float:
        ab = a.get("_orig_bbox") or a.get("bbox") or []
        bb = b.get("_orig_bbox") or b.get("bbox") or []
        if len(ab) != 4 or len(bb) != 4:
            return 0.0
        ax1, _, ax2, _ = ab
        bx1, _, bx2, _ = bb
        inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        smaller = max(1.0, min(ax2 - ax1, bx2 - bx1))
        return inter / smaller

    n = len(pictures)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _x_overlap(pictures[i], pictures[j]) >= 0.5:
                _union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(_find(i), []).append(i)

    # For each cluster, decide the consensus column. Strategy:
    #
    # 1. Tier-weighted voting from EVERY picture's full candidate list (not
    #    just its winner). An empty cell (tier 0) anywhere in any sibling's
    #    candidate set is worth more than many tier-1 numeric-cell votes —
    #    a single empty cell in the cluster's row range is the cleanest
    #    "this is the picture column" signal, and survives translation.
    # 2. Among the picture's own candidates, restrict to that consensus
    #    column. If the picture has no candidate in that column (rare —
    #    only when the cluster spans multiple distinct image columns),
    #    fall back to the picture's individual winner.
    TIER_WEIGHT = {0: 100, 1: 1, 2: 0.01}

    for members in clusters.values():
        col_score: Dict[int, float] = {}
        for idx in members:
            for (_ar, c, tier, dist) in individual[idx][2]:
                # Closer cells contribute more even within the same tier,
                # so adjacent-column ties resolve toward the picture's
                # actual centroid.
                proximity = 1.0 / (1.0 + dist / max(1.0, tbl_w_px / max_cols))
                col_score[c] = col_score.get(c, 0.0) + TIER_WEIGHT.get(tier, 0.01) * proximity
        if not col_score:
            continue
        consensus_col = max(col_score.keys(), key=lambda c: (col_score[c], -c))

        for idx in members:
            anchor = individual[idx][1]
            if anchor is None:
                continue
            anchor_r, _orig_c = anchor
            if (anchor_r, consensus_col) in cell_anchors:
                final = (anchor_r, consensus_col)
            else:
                # Find a row anchor in the consensus column whose vertical
                # span covers this picture's row.
                final = None
                for (ar, ac), (_txt, _cs, rs, _h) in cell_anchors.items():
                    if ac != consensus_col:
                        continue
                    if ar <= anchor_r < ar + max(1, rs):
                        final = (ar, ac)
                        break
                if final is None:
                    final = anchor
            out.setdefault(final, []).append(pictures[idx])
    return out


def compute_col_weights(
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    max_cols: int,
) -> List[int]:
    """Per-column weights inferred from cell text.

    Two signals combine: `_display_width` (full content length, drives the
    'bigger column for more text' bias) and `_longest_token_width` (longest
    unbreakable Latin/digit run, which sets a floor so a column holding a
    long word/code doesn't get squeezed below that word's width).
    """
    col_weight = [1] * max_cols
    col_min = [1] * max_cols
    for (_r, c), (txt, cs, _rs, _h) in cell_anchors.items():
        if cs == 1:
            col_weight[c] = max(col_weight[c], _display_width(txt))
            col_min[c] = max(col_min[c], _longest_token_width(txt))
        else:
            share = max(1, _display_width(txt) // cs)
            tok_share = max(1, _longest_token_width(txt) // cs)
            for cc in range(c, c + cs):
                col_weight[cc] = max(col_weight[cc], share)
                col_min[cc] = max(col_min[cc], tok_share)
    # Lift each column's weight so the longest-unbreakable token always
    # has at least that much weight — keeps narrow-but-text-bearing columns
    # readable (e.g. an 'N/ACC' column was being squeezed to ~5mm).
    return [max(w, m) for w, m in zip(col_weight, col_min)]


def _inherit_header_colspans(
    cell_anchors: Dict[Tuple[int, int], Tuple[str, int, int, bool]],
    occupied: List[List[bool]],
    max_cols: int,
    n_rows: int,
) -> None:
    """Make body rows inherit the table's column grouping.

    Many source tables group several columns under one spanning header (e.g. a
    'Torque de falha' header with colspan=2 over 'Valor' + 'Modo' sub-columns).
    Summary rows ('Total', 'Average', 'Média', ...) then carry a single value
    under that whole group and leave the other grouped columns blank. OCR
    commonly emits those as a value cell plus a SEPARATE empty cell rather than
    one `colspan` cell, so the value lands in one sub-column instead of
    spanning the group as it does in the source.

    This pass detects each colspan group and, for every other row in which
    exactly ONE column of the group carries text and the rest are empty,
    replaces those cells with a single value cell spanning the whole group.
    Rows that fill more than one grouped column (real data rows) are left
    untouched. Mutates `cell_anchors`/`occupied` in place.

    General by construction — it keys only on:
      * the presence of a colspan>1 cell (the group), and
      * per-row emptiness within that group,
    with no table-, column-, document-, or language-specific assumptions. It
    also tolerates headers emitted as plain <td> rows (no <thead>) and values
    that sit in any sub-column of the group, not just the leftmost.
    """
    # 1) Column groups = ranges spanned by a colspan>1 cell. Prefer header
    #    cells; if the table marked no header, fall back to the first row,
    #    which is where a grouping header almost always sits.
    spanning = [
        (c, c + cs)
        for (r, c), (txt, cs, rs, is_h) in cell_anchors.items()
        if is_h and cs > 1
    ]
    if not spanning:
        spanning = [
            (c, c + cs)
            for (r, c), (txt, cs, rs, is_h) in cell_anchors.items()
            if r == 0 and cs > 1
        ]
    if not spanning:
        return
    # Dedup + drop nested/overlapping (keep the widest, leftmost-first).
    groups: List[Tuple[int, int]] = []
    for a, b in sorted(set(spanning), key=lambda g: (g[0], -(g[1] - g[0]))):
        if groups and a < groups[-1][1]:
            continue
        groups.append((a, b))

    for r in range(n_rows):
        for (a, b) in groups:
            present = [
                (c, cell_anchors[(r, c)])
                for c in range(a, b)
                if (r, c) in cell_anchors
            ]
            # Need every grouped column present as its own cell (cs==1) — skip
            # when a rowspan from above already occupies part of the group, or
            # when the group itself is the spanning header cell.
            if len(present) != (b - a) or any(v[1] != 1 for _c, v in present):
                continue
            nonempty = [(c, v) for (c, v) in present if (v[0] or "").strip()]
            if len(nonempty) != 1:
                continue  # 0 = leave blank; 2+ = real data row, leave split
            _val_c, val_v = nonempty[0]
            t, _cs, rs, is_h = val_v
            # Collapse the group into one value cell at the group start,
            # spanning the full width. Covered columns stay `occupied`, so the
            # render loop emits them as gridSpan continuation (skips them).
            for c, _v in present:
                del cell_anchors[(r, c)]
            cell_anchors[(r, a)] = (t, b - a, rs, is_h)
            for c in range(a, b):
                occupied[r][c] = True


def render_table(
    ctx,
    entry: Dict,
    pictures_for_table: Optional[List[Dict]] = None,
    orig_bbox: Optional[List[int]] = None,
) -> None:
    text = (entry.get("text") or "").strip()
    if not text:
        return
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    # Reflow may have shifted/grown this table's bbox downward while the
    # contained pictures stayed frozen at their original coordinates. Picture
    # → cell assignment must therefore be computed in the PRE-reflow frame
    # (the frame the picture coords still live in); using the post-reflow
    # bbox would map the first picture into the header row. Cell sizing
    # (column widths, row heights) still uses the live `bbox`.
    assign_bbox = orig_bbox if (orig_bbox and len(orig_bbox) == 4) else bbox
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
        # Fall back to text rendering inside a positioned box. Pin the bbox
        # (`body_auto_fit=False`) so an oversize translation can't grow past
        # the source rectangle; `fit_multiline` will shrink-or-truncate so the
        # rendered content always fits.
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
                body_auto_fit=False,
            )
        )
        return

    max_cols, n_rows, cell_anchors, occupied = parse_table_grid(rows)

    # Make summary/total rows inherit the header's column grouping so a single
    # value under a multi-column header spans that group (as in the source)
    # instead of sitting in one sub-column with blank cells beside it.
    _inherit_header_colspans(cell_anchors, occupied, max_cols, n_rows)

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
    if pictures_for_table:
        pic_assignments = _assign_pictures_to_cells(
            pictures_for_table, assign_bbox, col_weight, cell_anchors,
            max_cols, n_rows,
        )
        for (_r, c), pics in pic_assignments.items():
            for pic in pics:
                pb = pic.get("_orig_bbox") or pic.get("bbox") or []
                if len(pb) != 4:
                    continue
                pic_w_px = max(1.0, pb[2] - pb[0])
                # ~5 CJK chars per 11pt column ≈ 1 char per 2.2 pt. At
                # zoom=2 that's 1 char per ~4.4 px.
                pic_weight = max(1, int(pic_w_px / 4.4))
                col_weight[c] = max(col_weight[c], pic_weight)

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
    min_row_pt = natural_h_px * 1.06
    row_h_pts = [max(rh, min_row_pt) for rh in row_h_pts]

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
        for col_idx in range(max_cols):
            anchor = cell_anchors.get((row_idx, col_idx))
            if anchor is not None:
                cell_text, cs, rs, is_header = anchor
                cell_w_emu = sum(col_w_emus[col_idx:col_idx + cs])
                # Per-cell fit. Translation expansion is non-uniform — one
                # cell may become 8 chars while its neighbour becomes 80 —
                # so each cell is re-fitted against (cell_w, row_h). The
                # base size is `cell_size_pt` (the table-wide cap derived
                # from the narrowest column); `fit_multiline` shrinks
                # further inside that budget and truncates with `…` when
                # even the floor doesn't fit, so cell content never
                # overflows the fixed row height (`hRule="exact"`).
                cell_h_pt_for_fit = sum(
                    row_h_pts[row_idx + dr] for dr in range(max(1, rs))
                )
                cell_w_pt_for_fit = cell_w_emu / EMU_PER_PT
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
                pics_here = pic_inline_by_cell.get((row_idx, col_idx)) or []
                pic_paragraphs = []
                cell_h_emu = int(round(row_h_pts[row_idx] * EMU_PER_PT))
                if rs > 1:
                    cell_h_emu = sum(
                        int(round(row_h_pts[row_idx + dr] * EMU_PER_PT))
                        for dr in range(rs)
                    )
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
                    # 1 px → 1 pt → EMU_PER_PT EMU (matches the 72-dpi
                    # assumption used everywhere else for bbox math).
                    intrinsic_w_emu = iw * EMU_PER_PT
                    intrinsic_h_emu = ih * EMU_PER_PT
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
                r = row_idx - 1
                while r >= 0:
                    if (r, col_idx) in cell_anchors:
                        _t, _cs, _rs, _h = cell_anchors[(r, col_idx)]
                        if _rs > 1 and r + _rs > row_idx:
                            above_is_vmerge = True
                        break
                    if not occupied[r][col_idx]:
                        break
                    r -= 1
                if above_is_vmerge:
                    empty_para = build_paragraph_xml("", alignment="center")
                    cells_xml.append(
                        "<w:tc>"
                        f'<w:tcPr><w:tcW w:w="{col_w_emus[col_idx]}" w:type="dxa"/>'
                        '<w:vMerge w:val="continue"/>'
                        '<w:vAlign w:val="center"/></w:tcPr>'
                        f"{empty_para}"
                        "</w:tc>"
                    )
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