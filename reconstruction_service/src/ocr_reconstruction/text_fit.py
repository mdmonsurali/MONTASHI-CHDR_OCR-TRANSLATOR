"""Text measurement + binary-search font fit. CJK-aware wrapping that matches
how Word lays text out closely enough for fitting.

Multi-line fitting note
-----------------------
Text boxes use `<a:noAutofit/>` with a fixed `cy`, so Word hard-clips any
content that runs past the box height. Pillow measurement and Word layout
disagree slightly on line height, and that error accumulates per line, which
is why a long block's last line used to render half-clipped. We now make
measurement equal render by construction:

  1. `fit_multiline` returns the exact wrap it measured.
  2. That wrap is frozen into hard `<w:br/>` breaks so Word renders the same
     line count we measured (no re-wrapping with a substituted font).
  3. Every paragraph zeroes before/after spacing, and multi-line text boxes
     pin `lineRule="exact"` at `box_height / n_lines`, so N lines occupy
     exactly the box height. Vertical overflow past the clip boundary becomes
     impossible regardless of which font Word substitutes.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from PIL import ImageDraw, ImageFont


_FONT_FALLBACK_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


def _resolve_font_path() -> Optional[str]:
    for p in _FONT_FALLBACK_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


_FONT_PATH = _resolve_font_path()
# Reuse a single ImageDraw for measurement so we don't reallocate per call.
_MEASURE_DRAW = ImageDraw.Draw(
    __import__("PIL.Image", fromlist=["new"]).new("RGB", (1, 1))
)
_FONT_CACHE: Dict[Tuple[str, int], "ImageFont.FreeTypeFont"] = {}


def get_font(size_px: int) -> Optional["ImageFont.FreeTypeFont"]:
    if not _FONT_PATH or size_px <= 0:
        return None
    key = (_FONT_PATH, size_px)
    f = _FONT_CACHE.get(key)
    if f is None:
        try:
            f = ImageFont.truetype(_FONT_PATH, size_px)
        except Exception:
            return None
        _FONT_CACHE[key] = f
    return f


# CJK / fullwidth ranges. Han, Hiragana/Katakana, Hangul, CJK punctuation and
# the fullwidth forms block (（）、，。： etc.). These characters carry no spaces
# and Word may break a line between any two of them.
_CJK_RE = re.compile(
    r"[ᄀ-ᇿ⺀-⻿　-〿぀-ヿ㄰-㆏"
    r"㐀-䶿一-鿿ꀀ-꓏가-힯豈-﫿"
    r"︰-﹏＀-￯]"
)


def has_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))


def is_cjk_char(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def measure_width_px(text: str, font, size_px: int) -> float:
    """Width of `text` in px, CJK-aware.

    The fallback measuring font (DejaVuSans) has no CJK glyphs, so measuring a
    Chinese string with it under-counts width by ~40% and wrecks the line
    count. CJK ideographs and fullwidth punctuation are full-width (≈ 1 em)
    in essentially every font, so we count them as `size_px` each and measure
    only the non-CJK runs with the real font. This is font-independent and
    matches how Word lays CJK out closely enough for fitting."""
    if not text:
        return 0.0
    total = 0.0
    run = ""
    for ch in text:
        if is_cjk_char(ch):
            if run:
                b = _MEASURE_DRAW.textbbox((0, 0), run, font=font)
                total += (b[2] - b[0])
                run = ""
            total += size_px            # full-width em
        else:
            run += ch
    if run:
        b = _MEASURE_DRAW.textbbox((0, 0), run, font=font)
        total += (b[2] - b[0])
    return total


def _segments(paragraph: str) -> List[str]:
    """Split a paragraph into break-able tokens. A break is allowed between any
    two tokens. Latin words stay whole (break only at spaces); each CJK char is
    its own token (break allowed between characters, as Word does). Trailing
    spaces stay attached to the preceding token so the break point is correct."""
    toks: List[str] = []
    buf = ""
    for ch in paragraph:
        if is_cjk_char(ch):
            if buf:
                toks.append(buf)
                buf = ""
            toks.append(ch)
        elif ch == " ":
            buf += " "
            toks.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        toks.append(buf)
    return toks


def wrap_to_width(text: str, font, max_w_px: float, size_px: int) -> List[str]:
    """Greedy wrap that respects existing newlines AND breaks between CJK
    characters (not just at spaces), measuring width CJK-aware."""
    lines: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        cur = ""
        for tok in _segments(paragraph):
            candidate = cur + tok
            if not cur.strip() or measure_width_px(candidate, font, size_px) <= max_w_px:
                cur = candidate
            else:
                lines.append(cur.rstrip())
                cur = "" if tok.strip() == "" else tok
        if cur.strip():
            lines.append(cur.rstrip())
    return lines or [""]


def _wrapped_lines_if_fit(
    text: str,
    size_px: int,
    box_w_px: float,
    box_h_px: float,
    width_safety: float = 0.97,   # wrap a touch early so Word never adds a line
    height_safety: float = 1.06,  # leave ~6% vertical headroom for font substitution
) -> Optional[List[str]]:
    """Return the wrapped lines if `text` fits the box at `size_px`, else None.

    The measuring font (DejaVuSans) is wider than most fonts Word substitutes,
    and `width_safety` wraps slightly early, so the line count we return is an
    UPPER bound on what Word will produce — Word never adds a line we didn't
    account for. `height_safety` keeps the natural glyph height comfortably
    below the per-line slot we later pin, so `lineRule="exact"` can't clip
    ascenders/descenders.
    """
    font = get_font(size_px)
    if font is None:
        return None
    lines = wrap_to_width(text, font, box_w_px * width_safety, size_px)
    # Guard against a single un-wrappable long token overflowing the width.
    for line in lines:
        if measure_width_px(line, font, size_px) > box_w_px:
            return None
    asc, desc = font.getmetrics()
    # CJK glyphs sit in a taller (~1.2 em) line box than the Latin ascent+
    # descent of the fallback font reports, so take the larger of the two.
    natural_h = asc + desc
    if has_cjk(text):
        natural_h = max(natural_h, size_px * 1.2)
    line_h = natural_h * height_safety
    if line_h * len(lines) > box_h_px:
        return None
    return lines


def fit_multiline(
    text: str,
    box_w_pt: float,
    box_h_pt: float,
    max_size_pt: float = 11.0,
    min_size_pt: float = 5.0,
    dpi: int = 72,
) -> Tuple[float, Optional[List[str]]]:
    """Largest font size (pt) at which `text` word-wraps to fit a
    (box_w_pt × box_h_pt) bbox, plus the exact wrapped lines at that size.

    Returns:
        (size_pt, lines)
        - `lines` is the frozen wrap the caller should render (joined with
          newlines → hard breaks). It is None only when measurement is
          impossible (no usable font installed); the caller then renders the
          raw text and Word wraps on its own.
        - When nothing fits even at `min_size_pt`, returns
          (min_size_pt, <wrap at min_size>) so the caller can still pin an
          exact line height and keep every line at least partially visible.

    Binary search runs over integer pixel sizes for stability. dpi=72 because
    1 pt = 1 px at 72 dpi, matching how Word converts pt to render units.
    """
    if not text.strip() or box_w_pt <= 0 or box_h_pt <= 0 or _FONT_PATH is None:
        return max_size_pt, None

    pt_to_px = dpi / 72.0
    box_w_px = box_w_pt * pt_to_px
    box_h_px = box_h_pt * pt_to_px
    hi = max(min_size_pt, max_size_pt)
    lo = min_size_pt

    # Try hi first — most boxes already fit at the OCR-detected size.
    hi_lines = _wrapped_lines_if_fit(text, int(round(hi * pt_to_px)), box_w_px, box_h_px)
    if hi_lines is not None:
        return float(hi), hi_lines

    best_size = float(min_size_pt)
    best_lines: Optional[List[str]] = None
    while lo <= hi:
        mid = (lo + hi) / 2.0
        size_px = max(1, int(round(mid * pt_to_px)))
        lines = _wrapped_lines_if_fit(text, size_px, box_w_px, box_h_px)
        if lines is not None:
            best_size, best_lines = mid, lines
            lo = mid + 0.5
        else:
            hi = mid - 0.5

    if best_lines is None:
        # Nothing fit, even at min — freeze a wrap at the floor size so the
        # caller can still pin an exact line height across the bbox. Some
        # glyph clipping is unavoidable here, but it stays bounded.
        floor_px = max(1, int(round(min_size_pt * pt_to_px)))
        floor_font = get_font(floor_px)
        if floor_font is not None:
            best_lines = wrap_to_width(text, floor_font, box_w_px * 0.97, floor_px)

    return max(float(min_size_pt), best_size), best_lines
